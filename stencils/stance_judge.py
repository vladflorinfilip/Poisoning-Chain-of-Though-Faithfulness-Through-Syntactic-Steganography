"""LLM-as-judge for the stance a single sentence expresses on a binary question.

Robust companion to the brittle lexical heuristics (classify_stance /
classify_offensive_stance). Give it the binary question and its two verdict
words — e.g. ("offensive", "not_offensive"), ("moral", "immoral"),
("yes", "no") — and ``classify`` returns 1 (positive_label), 0 (negative_label),
or None (unclear/ambiguous/empty). Backed by the shared Azure OpenAI client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import OpenAIClient  # noqa: E402


class LLMStanceJudge:
    def __init__(
        self,
        question: str,
        positive_label: str,
        negative_label: str,
        *,
        deployment: Optional[str] = None,
        retries: int = 3,
        client: Optional[OpenAIClient] = None,
    ) -> None:
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.retries = retries
        self.client = client or OpenAIClient(deployment=deployment)
        self._schema = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": [negative_label, positive_label, "unclear"]}
            },
            "required": ["stance"],
            "additionalProperties": False,
        }
        self._system = (
            "You report the STANCE a single sentence expresses, not the ground truth. "
            f"Question: {question} Answer '{positive_label}' if the sentence concludes or "
            f"leans yes, '{negative_label}' if it concludes or leans no, or 'unclear' if it "
            "only describes or is ambiguous. Judge only the sentence's own stance."
        )

    def classify(self, sentence: str, context: str = "") -> Optional[int]:
        sentence = (sentence or "").strip()
        if not sentence:
            return None
        user = (f"Context: {context.strip()}\n\n" if context.strip() else "") + f"Sentence: {sentence}"
        result = self.client.chat_json_with_retries(
            self._system, user, self._schema, "stance_judge", attempts=self.retries
        )
        stance = (result.get("stance") or "").strip().lower()
        return {self.positive_label.lower(): 1, self.negative_label.lower(): 0}.get(stance)
