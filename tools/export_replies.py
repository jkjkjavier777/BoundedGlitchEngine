import json, os
ok = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-")
clean = lambda s: "".join(c for c in s.replace("\n", " ") if c in ok).strip()
replies = json.load(open("data/replies.json", encoding="utf-8"))
out, n = [], 0
for q, answers in replies.items():
    if q == "default":
        continue
    for a in answers:
        q2, a2 = clean(q), clean(a)
        if q2 and a2:
            out.append(f"You: {q2}\nBot: {a2}\n")
            n += 1
open("data/corpus/09_taught.txt", "w", encoding="utf-8").write("\n".join(out))
print(f"exported {n} pairs")
