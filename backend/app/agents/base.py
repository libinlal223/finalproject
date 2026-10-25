"""
app/agents/base.py
──────────────────
Abstract base class and utilities for all VADER debate agents.

Responsibilities:
  - Coordinate with the configured BaseLLMClient.
  - Wrap untrusted user inputs with clear delimiter boundaries to resist injection.
  - Enforce strict JSON output parsing and Pydantic validation.
  - Translate malformed outputs and provider errors predictably into LLMError subclasses.
  - Prevent private chain-of-thought or secrets from leaking to callers or logs.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm.base import (
    BaseLLMClient,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMResponse,
    LLMValidationError,
    build_llm_client,
    sanitize_error_message,
)
from app.models.schemas import AgentRole

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base agent encapsulating client coordination and structured parsing.
    """

    role: AgentRole
    system_prompt: str

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize the agent.

        Parameters
        ----------
        client:
            Optional BaseLLMClient. If None, the client is resolved dynamically
            from the active environment settings via build_llm_client().
        config:
            Optional LLMConfig override. If None, provider defaults are used.
        """
        self._config = config or self.default_config()
        if client is not None:
            self._client = client
        else:
            settings = get_settings()
            self._client = build_llm_client(settings.llm_provider.value, self._config)

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @classmethod
    def default_config(cls) -> LLMConfig:
        """Return the default LLMConfig for this agent role."""
        return LLMConfig(temperature=0.3, max_tokens=2048)

    @property
    def client(self) -> BaseLLMClient:
        """Return the underlying LLM client."""
        return self._client

    @property
    def config(self) -> LLMConfig:
        """Return the agent's LLM configuration."""
        return self._config

    def _wrap_untrusted_input(self, text: str) -> str:
        """
        Delimit untrusted user inputs to protect against prompt injection.
        """
        clean = text.strip()
        return f"<user_query>\n{clean}\n</user_query>"

    def _parse_json_response(self, response: LLMResponse, schema: type[T]) -> T:
        """
        Extract and validate structured JSON output from the LLM response.

        Raises LLMValidationError if JSON is malformed or does not conform
        to the expected Pydantic schema.
        """
        raw = response.content.strip()
        if not raw:
            raise LLMValidationError(
                f"Agent '{self.role.value}' returned an empty response."
            )

        # 1. Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        cleaned = raw
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            # If no fences, attempt to locate the outermost JSON object
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                cleaned = json_match.group(0).strip()

        # 2. Attempt JSON deserialization
        try:
            data = json.loads(cleaned, strict=False)
        except json.JSONDecodeError as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.warning(
                "Agent '%s' output failed JSON decoding: %s",
                self.role.value,
                sanitized,
            )
            raise LLMValidationError(
                f"Agent '{self.role.value}' output is not valid JSON: {sanitized}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMValidationError(
                f"Agent '{self.role.value}' expected a JSON object but got {type(data).__name__}."
            )

        # 3. Validate against Pydantic schema
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.warning(
                "Agent '%s' schema validation failed: %s",
                self.role.value,
                sanitized,
            )
            raise LLMValidationError(
                f"Agent '{self.role.value}' output failed schema validation: {sanitized}"
            ) from exc

    async def _execute_prompt(
        self,
        user_prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> tuple[T, LLMResponse]:
        """
        Send system and user messages to the client and return validated structured output.
        """
        active_system = system_prompt or self.system_prompt
        messages = [
            LLMMessage(role="system", content=active_system),
            LLMMessage(role="user", content=user_prompt),
        ]

        self._logger.debug("Executing prompt for agent '%s'", self.role.value)
        try:
            response = await self._client.complete(messages)
        except LLMError:
            # Re-raise known application-level LLM errors directly
            raise
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            raise LLMError(
                f"Agent '{self.role.value}' encountered an unexpected error: {sanitized}"
            ) from exc

        parsed = self._parse_json_response(response, schema)
        return parsed, response
