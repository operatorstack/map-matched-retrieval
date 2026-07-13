from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_REWRITE_PROMPT_VERSION = "cast-standalone-v1"

_SYSTEM_INSTRUCTION = """\
Rewrite the current conversational search utterance as one concise, standalone search query.
Resolve references and omitted context using only the prior user utterances.
Preserve the current information need and do not answer it.
Return only the rewritten query with no explanation, label, quotation marks, or markdown."""


class GeminiDependencyUnavailableError(ImportError):
    pass


@dataclass(frozen=True, slots=True)
class GeminiRewriteConfig:
    model: str = DEFAULT_GEMINI_MODEL
    temperature: float = 0.0


class GeminiQueryRewriter:
    def __init__(
        self,
        *,
        generate_content: Callable[..., object],
        config: GeminiRewriteConfig,
    ) -> None:
        self._generate_content = generate_content
        self.config = config

    def rewrite(self, *, history: Sequence[str], query: str) -> str:
        request = {
            "prior_user_utterances": list(history),
            "current_utterance": query,
        }
        response = self._generate_content(
            model=self.config.model,
            contents=f"{_SYSTEM_INSTRUCTION}\n\nConversation:\n{json.dumps(request, ensure_ascii=False)}",
            config={"temperature": self.config.temperature},
        )
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("Gemini returned no rewritten query")
        return response_text.strip()


def create_gemini_query_rewriter(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> GeminiQueryRewriter:
    effective_api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "GEMINI_API_KEY must be set when the Gemini rewrite baseline is enabled"
        )
    try:
        genai = importlib.import_module("google.genai")
    except ImportError as error:
        raise GeminiDependencyUnavailableError(
            "Gemini rewrite requires the 'gemini' extra: "
            "pip install 'map-matched-retrieval[eval,gemini]'"
        ) from error
    client_factory = getattr(genai, "Client", None)
    if not callable(client_factory):
        raise GeminiDependencyUnavailableError("google.genai.Client is unavailable")
    client = client_factory(api_key=effective_api_key)
    models = getattr(client, "models", None)
    generate_content = getattr(models, "generate_content", None)
    if not callable(generate_content):
        raise GeminiDependencyUnavailableError(
            "google.genai Client does not provide models.generate_content"
        )
    return GeminiQueryRewriter(
        generate_content=generate_content,
        config=GeminiRewriteConfig(model=model),
    )
