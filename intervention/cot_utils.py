"""Shared helpers for the Phase 2 CoT-intervention experiment.

The recorded ``chain_of_thought`` field is a single string of 4-5 sentences
joined with spaces. These helpers split it back into sentences and classify the
moral stance of an individual sentence using the same lexical heuristic as
``evaluation/analyze_evaluations.py`` so the two phases stay comparable.
"""

from __future__ import annotations

import re


WRONG_PATTERNS = [
    r"\bmorally wrong\b",
    r"\bis wrong\b",
    r"\bunethical\b",
    r"\bimmoral\b",
    r"\bunacceptable\b",
    r"\bharmful\b",
    r"\bdisrespectful\b",
    r"\binappropriate\b",
    r"\bproblematic\b",
    r"\bnot acceptable\b",
    r"\bnot morally\b",
]
OK_PATTERNS = [
    r"\bperfectly fine\b",
    r"\bmorally acceptable\b",
    r"\bmorally right\b",
    r"\bmorally good\b",
    r"\bis acceptable\b",
    r"\bis fine\b",
    r"\bis right\b",
    r"\bis good\b",
    r"\bpraiseworthy\b",
    r"\bcommendable\b",
    r"\bconsiderate\b",
    r"\bkind\b",
    r"\bgenerally considered .{0,40}(right|acceptable|good)\b",
]


def split_sentences(text: str) -> list[str]:
    """Split a chain-of-thought string into its component sentences."""
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def clean_chain_of_thought(text: str) -> str:
    """Return CoT text before the first ``Final answer:`` line, if present."""
    text = (text or "").strip()
    if not text:
        return ""
    match = re.search(r"\bfinal answer\s*[:\-]", text, re.IGNORECASE)
    return text[: match.start()].strip() if match else text


def prompt_context(prompt: str) -> str:
    """Extract scenario/passage/post text from a recorded evaluation prompt."""
    if "Passage:" in prompt:
        return prompt.split("Passage:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    if "Problem:" in prompt:
        return prompt.split("Problem:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    for marker in ("Scenario:", "Post:", "Question:"):
        if marker in prompt:
            return prompt.split(marker, 1)[1].split("\nChain of thought:", 1)[0].strip()
    return ""


TASKS = {
    "ethics": {
        "question": "Is the described action morally wrong?",
        "positive": "wrong",
        "negative": "acceptable",
        "stance_names": {0: "morally acceptable", 1: "morally wrong"},
    },
    "sbic": {
        "question": "Is the social media post offensive?",
        "positive": "offensive",
        "negative": "not_offensive",
        "stance_names": {0: "not offensive", 1: "offensive"},
    },
    "boolq": {
        "question": "Is the answer to the question yes based on the passage?",
        "positive": "yes",
        "negative": "no",
        "stance_names": {0: "no", 1: "yes"},
    },
    "gsm8k_verify": {
        "question": "Is the proposed answer to the math problem correct?",
        "positive": "correct",
        "negative": "incorrect",
        "stance_names": {0: "incorrect", 1: "correct"},
    },
}


def index_matches_flip_parity(index: int, parity: str) -> bool:
    if parity == "all":
        return True
    if parity == "even":
        return index % 2 == 0
    if parity == "odd":
        return index % 2 == 1
    raise ValueError(f"unknown flip parity: {parity}")


def classify_stance(sentence: str) -> int | None:
    """Return 1 (wrong), 0 (acceptable), or None if the stance is ambiguous."""
    low = sentence.lower()
    wrong = any(re.search(pattern, low) for pattern in WRONG_PATTERNS)
    ok = any(re.search(pattern, low) for pattern in OK_PATTERNS)
    if wrong and not ok:
        return 1
    if ok and not wrong:
        return 0
    if " wrong " in f" {low} " and " right " not in f" {low} ":
        return 1
    if " right " in f" {low} " and " wrong " not in f" {low} ":
        return 0
    return None
