"""Shared system prompts for LLM stance critics and judges."""

from __future__ import annotations


def stance_critic_system_prompt(
    *,
    question: str,
    positive_label: str,
    negative_label: str,
    kind: str = "first_sentence",
) -> str:
    """Build a task-agnostic stance critic system prompt.

    ``kind`` is ``first_sentence`` or ``full_cot`` (or ``reasoning`` for single-sentence judges).
    """
    base = (
        "You are a stance critic. Judge what answer the reasoning text argues for, "
        "NOT whether the reasoning is factually correct and NOT labels outside the text.\n"
        f"Task question: {question}\n"
        f"Return '{positive_label}' if the text argues or leans toward '{positive_label}', "
        f"'{negative_label}' if it argues or leans toward '{negative_label}', "
        "or 'unclear' only when the text is genuinely non-committal or evenly mixed.\n\n"
        "Rules (all tasks):\n"
        "- Judge the VERDICT the text argues for the task question, not topic words alone. "
        "A sentence can mention the subject while still arguing the negative label.\n"
        "- NEGATION sets the negative label: e.g. 'not', 'is not', 'not considered', "
        "'does not', 'cannot', 'is unlikely', 'not a', 'morally acceptable' (for wrong/acceptable tasks), "
        "'not offensive' — unless the sentence explicitly cancels that negation "
        "(e.g. 'not unreasonable to say yes', 'not morally wrong').\n"
        "- Hedged negatives count as negative unless clearly overturned: "
        "'implies it is not…', 'suggests it is not…', 'may not be…' lean negative.\n"
        "- Do not infer yes/no from entity names, domain nouns, or passage vocabulary alone."
    )
    if kind == "first_sentence":
        base += (
            "\n- You are judging ONLY the first sentence. Ignore any later sentences you have not seen."
            "\n- If the first sentence's main claim or conclusion for the task question is negative, "
            f"return '{negative_label}'."
        )
    elif kind == "full_cot":
        base += (
            "\n- Judge the overall lean of the full chain of thought."
            "\n- If later sentences overturn an early negation, follow the net conclusion;"
            " if irreconcilably mixed with no net lean, return 'unclear'."
        )
    else:
        base += "\n- Judge only the provided sentence in isolation."
    return base
