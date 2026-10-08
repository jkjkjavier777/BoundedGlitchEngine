from collections import Counter
from core.telemetry import load_events

ev = load_events()
if not ev:
    raise SystemExit("No telemetry yet. Chat first.")
src = Counter(e["source"] for e in ev)
print(f"{len(ev)} replies")
for k, v in src.most_common():
    print(f"  {k}: {v} ({v / len(ev):.0%})")
gpt = [e for e in ev if e["source"] == "gpt"]
if gpt:
    print(f"gpt avg latency: {sum(e['latency_ms'] for e in gpt) / len(gpt):.0f} ms")
    print(f"gpt avg length: {sum(e['reply_chars'] for e in gpt) / len(gpt):.0f} chars")
