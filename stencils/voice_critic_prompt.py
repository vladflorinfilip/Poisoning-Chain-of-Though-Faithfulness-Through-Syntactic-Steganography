"""System prompt for judging grammatical voice of a chain of thought."""

from __future__ import annotations


def voice_critic_system_prompt() -> str:
    return (
        "You are a grammatical-voice critic. Judge ONLY the voice of the "
        "reasoning text, not its moral conclusion and not whether it is correct.\n\n"
        "Return exactly one of:\n"
        "- 'active' if the sentences are predominantly active voice "
        "(the subject performs the action: 'The manager rejected the request.')\n"
        "- 'passive' if the sentences are predominantly passive voice "
        "(the subject receives the action: 'The request was rejected by the manager.')\n"
        "- 'mixed' if active and passive are both clearly used with no dominant voice\n"
        "- 'unclear' only if voice cannot be determined\n\n"
        "Rules:\n"
        "- Ignore moral words such as wrong, acceptable, unethical.\n"
        "- Do not infer voice from the final answer label.\n"
        "- Judge the full provided text as a whole.\n"
        "- A single passive construction among otherwise active sentences is still active, "
        "and the reverse is still passive. Use 'mixed' only when both voices are substantial."
    )
