import json
import time
from datetime import datetime, timezone
from config import DATA_DIR
import os

TELEMETRY_DIR = os.path.join(DATA_DIR, "telemetry")


def log_event(event):
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    event = {"ts": now.isoformat(), **event}
    path = os.path.join(TELEMETRY_DIR, now.strftime("%Y-%m-%d") + ".jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_events():
    events = []
    if not os.path.isdir(TELEMETRY_DIR):
        return events
    for name in sorted(os.listdir(TELEMETRY_DIR)):
        if name.endswith(".jsonl"):
            with open(os.path.join(TELEMETRY_DIR, name), encoding="utf-8") as f:
                events += [json.loads(line) for line in f if line.strip()]
    return events
