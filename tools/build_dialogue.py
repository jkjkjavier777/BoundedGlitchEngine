#!/usr/bin/env python3
"""Turn BoundedGlitchEngine's taught answers into USER:/GPT: training dialogue."""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path.home() / "The-BoundedGlitchGPT" / "data" / "00_bge_taught_dialogue.txt"
QW = {"what", "who", "how", "why", "when", "where", "which", "can", "do", "does", "is", "are"}
PREFIXES = ["hey, ", "so ", "quick question: ", "tell me, "]
rng = random.Random(0)


def clean(s):
    return " ".join(str(s).split())


def pairs():
    kb = ROOT / "data" / "knowledge.json"
    if kb.exists():
        for item in json.loads(kb.read_text(encoding="utf-8")):
            for a in item.get("a", []):
                yield clean(item["q"]), clean(a)
    rp = ROOT / "data" / "replies.json"
    if rp.exists():
        for q, answers in json.loads(rp.read_text(encoding="utf-8")).items():
            if q == "default":
                continue
            for a in answers:
                yield clean(q), clean(a)
    for f in sorted((ROOT / "data" / "corpus").glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^You:\s*(.+)\nBot:\s*(.+)$", text, re.M):
            yield clean(m.group(1)), clean(m.group(2))


def forms(q):
    base = q.rstrip("?.! ")
    if not base:
        return []
    cap = base[0].upper() + base[1:]
    if base.split()[0].lower() in QW:
        cap += "?"
    return [base, cap] + [p + base for p in rng.sample(PREFIXES, 2)]


def main():
    seen, blocks = set(), []
    for q, a in pairs():
        if not q or not a:
            continue
        for f in forms(q):
            if (f, a) in seen:
                continue
            seen.add((f, a))
            blocks.append(f"USER: {f}\nGPT: {a}\n")
    rng.shuffle(blocks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{len(blocks)} dialogue turns, {OUT.stat().st_size:,} bytes -> {OUT}")


if __name__ == "__main__":
    main()
