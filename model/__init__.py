"""KnowledgeModel: taught Q/A pairs, corpus index (TF-IDF), Markov chain."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "it", "this", "that", "for", "with", "as", "at", "by",
    "do", "does", "i", "you", "me", "my", "your", "what", "who", "how",
    "tell", "about", "please", "can", "could", "would",
}

SEED_KB = [
    {"q": "hello", "a": ["Hello. The archive is listening.",
                          "Hi. Ask, or teach me something."]},
    {"q": "who are you", "a": ["BoundedGlitchEngine. A small bot that learns what you teach it."]},
    {"q": "what is bge", "a": ["BGE is the BoundedGlitchEngine: taught answers, corpus search, Markov fallback."]},
    {"q": "how do i teach you", "a": ["Type: teach: your question = your answer"]},
]

SEED_CORPUS = """\
Knowledge is not a destination. It is a living ecosystem. Every question is a seed. Every answer is a branch. Every mistake is a root.

The Reward Shaping Engine governs how knowledge is retained. The net reward equals the base reward minus the penalty. Repeats are penalized and creativity is rewarded.

Continuity can be curator-maintained or autonomous. Curator-maintained continuity is rebuilt by a human through notes and prompts. Autonomous continuity would require a system to be shaped by its own prior state without outside help.

Chaos is not the enemy. It is the fuel that drives the archive forward. The archive is yours to explore, expand, and shape.
"""


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def norm(text):
    return " ".join(TOKEN.findall(text.lower()))


class KnowledgeModel:
    MARKOV_ORDER = 2

    def __init__(self, data_dir=None):
        root = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir else root / "data"
        self.corpus_dir = self.data_dir / "corpus"
        self.kb_path = self.data_dir / "knowledge.json"

        self.kb = []        # [{"q": str, "a": [str, ...]}]
        self.chunks = []    # corpus passages
        self.chain = defaultdict(list)
        self.docs, self.vecs, self.idf = [], [], {}

        self._seed()
        self._load_kb()
        self.train()

    # ---------------------------------------------------------- training
    def train(self):
        """Reload corpus files from disk; rebuild index and chain."""
        self.chunks = []
        files = sorted(p for p in self.corpus_dir.rglob("*")
                       if p.suffix.lower() in (".txt", ".md"))
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            self.chunks.extend(self._chunk(text))
        self._rebuild()
        return (f"Trained on {len(files)} file(s): {len(self.chunks)} passages, "
                f"{len(self.kb)} taught question(s), "
                f"{len(self.chain)} Markov states.")

    def teach(self, question, answer):
        key = norm(question)
        if not key:
            return "That question has no usable words."
        for entry in self.kb:
            if norm(entry["q"]) == key:
                if answer not in entry["a"]:
                    entry["a"].append(answer)
                n = len(entry["a"])
                break
        else:
            self.kb.append({"q": question.strip(), "a": [answer]})
            n = 1
        self._save_kb()
        self._rebuild()
        return f'Learned. "{question.strip()}" now has {n} answer(s).'

    def forget(self, question):
        key = norm(question)
        before = len(self.kb)
        self.kb = [e for e in self.kb if norm(e["q"]) != key]
        if len(self.kb) == before:
            return "Nothing like that in taught knowledge."
        self._save_kb()
        self._rebuild()
        return "Forgotten."

    def stats(self):
        return {
            "taught_questions": len(self.kb),
            "corpus_passages": len(self.chunks),
            "vocabulary": len(self.idf),
            "markov_states": len(self.chain),
        }

    # ------------------------------------------------------------ search
    def search(self, q_tokens):
        """Return (kb_score, kb_answers, corpus_score, corpus_text)."""
        qvec = self._vec(q_tokens)
        best = {"kb": (0.0, None), "corpus": (0.0, None)}
        for (kind, _, payload), dvec in zip(self.docs, self.vecs):
            score = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if score > best[kind][0]:
                best[kind] = (score, payload)
        return best["kb"][0], best["kb"][1], best["corpus"][0], best["corpus"][1]

    # ---------------------------------------------------------- internals
    def _seed(self):
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        if not self.kb_path.exists():
            self.kb = SEED_KB
            self._save_kb()
        if not any(self.corpus_dir.iterdir()):
            (self.corpus_dir / "starter.txt").write_text(SEED_CORPUS, encoding="utf-8")

    def _rebuild(self):
        self._reindex()
        self._build_chain()

    def _chunk(self, text, limit=350):
        chunks = []
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if len(para) < 20:
                continue
            buf = ""
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if buf and len(buf) + len(sent) > limit:
                    chunks.append(buf)
                    buf = ""
                buf = f"{buf} {sent}".strip()
            if buf:
                chunks.append(buf)
        return chunks

    def _reindex(self):
        self.docs = [("kb", e["q"], e["a"]) for e in self.kb]
        self.docs += [("corpus", c, c) for c in self.chunks]
        toks = [tokens(d[1]) for d in self.docs]
        df = Counter()
        for t in toks:
            df.update(set(t))
        n = len(self.docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vecs = [self._vec(t) for t in toks]

    def _vec(self, toks):
        v = {t: c * self.idf.get(t, 1.0) for t, c in Counter(toks).items()}
        length = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / length for t, x in v.items()}

    def _build_chain(self):
        self.chain = defaultdict(list)
        n = self.MARKOV_ORDER
        sources = list(self.chunks) + [a for e in self.kb for a in e["a"]]
        for text in sources:
            words = text.split()
            for i in range(len(words) - n):
                self.chain[tuple(words[i:i + n])].append(words[i + n])

    def _load_kb(self):
        try:
            data = json.loads(self.kb_path.read_text(encoding="utf-8"))
            self.kb = [{"q": e["q"], "a": [a for a in e["a"] if a]}
                       for e in data if e.get("q") and e.get("a")]
        except (OSError, ValueError, KeyError, TypeError):
            self.kb = []

    def _save_kb(self):
        tmp = self.kb_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.kb, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.kb_path)
