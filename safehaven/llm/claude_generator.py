"""Anthropic Claude API-based ResponseGenerator implementation."""

from __future__ import annotations

from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlock

from safehaven.models import ConversationContext


class ClaudeResponseGenerator:
    """Generate responses using the Anthropic Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, context: ConversationContext) -> str:
        """Call the Claude API and return the response text.

        Raises:
            ConnectionError: If the API is unreachable.
            ValueError: If context has no messages.
        """
        raw_messages = context.to_llm_messages()
        if not raw_messages:
            raise ValueError("ConversationContext contains no messages.")

        messages = cast(list[MessageParam], raw_messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": messages,
        }
        if context.system_prompt:
            kwargs["system"] = context.system_prompt

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIConnectionError as exc:
            raise ConnectionError(str(exc)) from exc

        first_block = response.content[0]
        if not isinstance(first_block, TextBlock):
            raise ValueError(f"Unexpected response block type: {first_block.type}")
        return first_block.text
