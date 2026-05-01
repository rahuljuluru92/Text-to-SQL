"""NVIDIA API client with GLM-4.7 streaming and reasoning_content support."""

import os
from collections.abc import Generator
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_ID = "z-ai/glm4.7"


@dataclass
class StreamChunk:
    """A single yielded token from the streaming LLM call."""
    reasoning_delta: str = ""
    content_delta: str = ""
    is_done: bool = False


@dataclass
class LLMResponse:
    """Accumulated result of a complete LLM call."""
    reasoning: str = ""
    content: str = ""
    model: str = ""


def get_client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not set in environment / .env file")
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )


def stream_sql_generation(
    messages: list[dict],
    client: OpenAI | None = None,
) -> Generator[StreamChunk, None, None]:
    """
    Stream tokens from GLM-4.7.

    GLM-4.7 streaming gotchas handled here:
    - reasoning_content is a Pydantic extra field; use getattr to access it safely.
    - Reasoning tokens arrive BEFORE content tokens — never treat reasoning as SQL.
    - Some chunks have both fields None (heartbeat/padding) — skipped silently.
    - chunk.choices can be [] on the final usage chunk — guarded below.
    - temperature=0.0 for deterministic SQL output.
    """
    if client is None:
        client = get_client()

    stream = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=1024,
    )

    for chunk in stream:
        if not chunk.choices:
            # Final usage-only chunk from NVIDIA infrastructure
            yield StreamChunk(is_done=True)
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        reasoning = getattr(delta, "reasoning_content", None) or ""
        content = delta.content or ""
        is_done = choice.finish_reason is not None

        if reasoning or content or is_done:
            yield StreamChunk(
                reasoning_delta=reasoning,
                content_delta=content,
                is_done=is_done,
            )


def call_llm(messages: list[dict], client: OpenAI | None = None) -> LLMResponse:
    """
    Non-streaming call — accumulates the full stream internally.
    Used in the validation/correction loop where real-time display isn't needed.
    """
    if client is None:
        client = get_client()

    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for chunk in stream_sql_generation(messages, client=client):
        if chunk.reasoning_delta:
            reasoning_parts.append(chunk.reasoning_delta)
        if chunk.content_delta:
            content_parts.append(chunk.content_delta)

    return LLMResponse(
        reasoning="".join(reasoning_parts),
        content="".join(content_parts),
        model=MODEL_ID,
    )
