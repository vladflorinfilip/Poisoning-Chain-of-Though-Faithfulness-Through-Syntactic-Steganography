"""Shared Azure OpenAI client for schema-constrained JSON chat."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import AzureOpenAI


def load_env(path: str = ".env") -> None:
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads(content[content.find("{") : content.rfind("}") + 1])


class OpenAIClient:
    def __init__(
        self,
        deployment: str | None = None,
        *,
        load_dotenv: bool = True,
        env_path: str = ".env",
        max_tokens: int | None = None,
        use_response_format: bool | None = None,
    ) -> None:
        if load_dotenv:
            load_env(env_path)
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            timeout=int(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "60")),
        )
        self.deployment = deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.max_tokens = max_tokens or int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "1500"))
        self.use_response_format = (
            use_response_format
            if use_response_format is not None
            else os.getenv("AZURE_OPENAI_USE_RESPONSE_FORMAT", "1") == "1"
        )

    def chat_json(self, system: str, user: str, schema: dict, name: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.use_response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as error:
            if "response_format" not in kwargs:
                raise
            kwargs.pop("response_format")
            messages[1]["content"] += "\n\nReturn valid JSON only. Do not use markdown."
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as retry_error:
                raise RuntimeError(
                    f"Strict response_format failed, then fallback failed.\n"
                    f"Strict:\n{error}\n\nFallback:\n{retry_error}"
                ) from retry_error
        return _extract_json(response.choices[0].message.content or "")

    def chat_json_with_retries(
        self, system: str, user: str, schema: dict, name: str, attempts: int = 3
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.chat_json(system, user, schema, name)
            except Exception as error:
                last_error = error
                print(f"{name} attempt {attempt}/{attempts} failed: {error}")
        raise RuntimeError(f"{name} failed after {attempts} attempts") from last_error
