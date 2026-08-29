from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def to_log(self) -> dict[str, int]:
        return {"in": self.input_tokens, "out": self.output_tokens}


@dataclass(frozen=True)
class StructuredResult(Generic[SchemaT]):
    value: SchemaT
    usage: TokenUsage


class StructuredClient(Protocol):
    def parse(
        self,
        schema: type[SchemaT],
        *,
        instructions: str,
        input_text: str,
    ) -> StructuredResult[SchemaT]: ...


class OpenAIStructuredClient:
    """Small Responses API adapter with bounded retry and exact token accounting."""

    def __init__(
        self,
        *,
        model: str,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        reasoning_effort: str = "low",
        client: object | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.max_retries = int(max_retries)
        self.retry_delay_s = float(retry_delay_s)
        self.reasoning_effort = reasoning_effort

    def parse(
        self,
        schema: type[SchemaT],
        *,
        instructions: str,
        input_text: str,
    ) -> StructuredResult[SchemaT]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=instructions,
                    input=input_text,
                    text_format=schema,
                    reasoning={"effort": self.reasoning_effort},
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError("model returned no parsed structured output")
                usage = getattr(response, "usage", None)
                return StructuredResult(
                    parsed,
                    TokenUsage(
                        int(getattr(usage, "input_tokens", 0) or 0),
                        int(getattr(usage, "output_tokens", 0) or 0),
                    ),
                )
            except Exception as exc:  # SDK exposes several transient subclasses.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * attempt)
        raise RuntimeError(f"structured model request failed after {self.max_retries} attempts") from last_error
