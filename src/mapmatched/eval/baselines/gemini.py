from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
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
    thinking_level: Literal["minimal", "low", "medium", "high"] = "minimal"
    maximum_attempts: int = 12
    initial_retry_delay: float = 5.0
    maximum_retry_delay: float = 60.0
    minimum_request_interval: float = 0.0


class GeminiQueryRewriter:
    def __init__(
        self,
        *,
        generate_content: Callable[..., object],
        config: GeminiRewriteConfig,
        client: object | None = None,
        cache_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if config.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be greater than zero")
        if config.minimum_request_interval < 0.0:
            raise ValueError("minimum_request_interval must be nonnegative")
        self._client = client
        self._generate_content = generate_content
        self._cache_path = cache_path
        self._cache = self._load_cache(cache_path)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_time: float | None = None
        self.config = config

    def rewrite(self, *, history: Sequence[str], query: str) -> str:
        request = {
            "prior_user_utterances": list(history),
            "current_utterance": query,
        }
        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\nConversation:\n{json.dumps(request, ensure_ascii=False)}"
        )
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "model": self.config.model,
                    "prompt_version": GEMINI_REWRITE_PROMPT_VERSION,
                    "request": request,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached_query = self._cache.get(cache_key)
        if cached_query is not None:
            return cached_query
        response = self._generate_with_retry(prompt)
        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("Gemini returned no rewritten query")
        rewritten_query = response_text.strip()
        self._cache[cache_key] = rewritten_query
        self._save_cache()
        return rewritten_query

    def _generate_with_retry(self, prompt: str) -> object:
        for attempt in range(self.config.maximum_attempts):
            try:
                self._wait_for_request_interval()
                return self._generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config={
                        "thinking_config": {
                            "thinking_level": self.config.thinking_level,
                        }
                    },
                )
            except Exception as error:
                final_attempt = attempt + 1 == self.config.maximum_attempts
                if final_attempt or not self._is_retryable(error):
                    raise
                delay = min(
                    self.config.initial_retry_delay * (2**attempt),
                    self.config.maximum_retry_delay,
                )
                self._sleep(delay)
        raise RuntimeError("Gemini retry loop ended without a response")

    def _wait_for_request_interval(self) -> None:
        now = self._monotonic()
        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            delay = self.config.minimum_request_interval - elapsed
            if delay > 0.0:
                self._sleep(delay)
        self._last_request_time = self._monotonic()

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600)

    @staticmethod
    def _load_cache(cache_path: Path | None) -> dict[str, str]:
        if cache_path is None or not cache_path.exists():
            return {}
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Gemini rewrite cache must contain a JSON object")
        cache: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Gemini rewrite cache keys and values must be strings")
            cache[key] = value
        return cache

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(f"{self._cache_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._cache_path)


def create_gemini_query_rewriter(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    cache_path: Path | None = None,
    minimum_request_interval: float = 0.0,
) -> GeminiQueryRewriter:
    effective_api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise ValueError("GEMINI_API_KEY must be set when the Gemini rewrite baseline is enabled")
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
    client = client_factory(
        api_key=effective_api_key,
        http_options={"timeout": 30_000},
    )
    models = getattr(client, "models", None)
    generate_content = getattr(models, "generate_content", None)
    if not callable(generate_content):
        raise GeminiDependencyUnavailableError(
            "google.genai Client does not provide models.generate_content"
        )
    return GeminiQueryRewriter(
        generate_content=generate_content,
        config=GeminiRewriteConfig(
            model=model,
            minimum_request_interval=minimum_request_interval,
        ),
        client=client,
        cache_path=cache_path,
    )
