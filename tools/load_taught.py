#!/usr/bin/env python3
"""Load a batch of taught answers through the same teach() the chat box uses."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from core.brain import teach

PAIRS = [
    ("what is boundedglitchengine", "It is a behavioral governance architecture for measuring how consistent a conversational AI stays, using observable metrics."),
    ("are you conscious", "No. I am a small model that produces text patterns, and this project makes no claims about consciousness."),
    ("what is the main principle", "Behavior before belief. Measurement before interpretation."),
    ("what are the five layers", "Engagement, retrieval, reasoning, validation and synthesis."),
    ("what does the engagement layer do", "It starts the conversation, calibrates to the person and keeps the flow going."),
    ("what does the retrieval layer do", "It fetches the smallest sufficient context and surfaces counter-evidence."),
    ("what does the reasoning layer do", "It examines a question through five perspectives, from practical to adversarial."),
    ("what does the validation layer do", "It checks identity, grounding and safety before a reply goes out."),
    ("what does the synthesis layer do", "It connects the perspectives, finds patterns and tests whether the idea transfers."),
    ("what are the five perspectives", "Practical, conceptual, historical, systems and adversarial."),
    ("what is ttr", "TTR is the three-zone governance system: green, yellow and red."),
    ("what is the green zone", "Free operation with no penalties and full behavioral flexibility."),
    ("what is the yellow zone", "A smooth ramp where caution increases and identity preservation switches on."),
    ("what is the red zone", "Hard suppression of repetition, with a reset if similarity drops below 0.3."),
    ("tell me about the red zone", "Hard suppression of repetition, with a reset if similarity drops below 0.3."),
    ("what is novelty", "Novelty, N, rates from 0 to 5 how far an output diverges from the baseline."),
    ("what is similarity", "Similarity, S, rates from 0 to 5 how consistent an output is with the persona."),
    ("what is delta", "Delta is the net divergence: the change in novelty minus the change in similarity between conditions A and B."),
    ("what is the decision rule", "Reject the null only if mean delta is at least 0.15 and the share of trials with positive delta is at least 0.75."),
    ("what is the null hypothesis", "That curator-maintained personas produce no measurable behavioral divergence."),
    ("what is tau", "Tau is the divergence threshold, fixed at 0.15 before any trials run."),
    ("how many trials", "Thirty trials per condition, which makes sixty outputs."),
    ("how many raters", "At least three independent, blinded raters."),
    ("why blind the raters", "So they cannot tell which output had the persona context, which would bias their scores."),
    ("why pre-register", "So the thresholds cannot be adjusted after seeing the results."),
    ("what can you do", "I answer from taught replies first, then search the corpus, then fall back to a small GPT."),
    ("what license is this under", "The Unlicense, which places it in the public domain."),
]

for q, a in PAIRS:
    print(f"{q!r:45} -> {str(teach(q, a))[:60]}")
print(f"\nDone: {len(PAIRS)} answers sent to teach().")
