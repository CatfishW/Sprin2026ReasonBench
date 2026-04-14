from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from reasonbench.clients.base import BaseLLMClient
from reasonbench.config import ClientConfig
from reasonbench.runtime.cache import SQLiteCache
from reasonbench.runtime.loop_detection import LoopDetectionConfig, detect_loop
from reasonbench.runtime.rate_limit import MinIntervalRateLimiter
from reasonbench.types import ChatMessage, GenerationRequest, GenerationResult


class OpenAICompatibleClient(BaseLLMClient):
    _RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: ClientConfig, cache: SQLiteCache | None = None):
        self.config = config
        self.cache = cache
        self._rate_limiter = MinIntervalRateLimiter(config.min_request_interval_s)
        self._session_local = threading.local()
        self._loop_config = LoopDetectionConfig(length_ceiling=config.loop_length_ceiling)
        self._chat_url = self._resolve_chat_url(config.base_url)

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            return base
        if base.endswith("/chat/completions") or base.endswith("/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def supports_batch(self) -> bool:
        return bool(self.config.supports_batch and self.config.completions_url)

    def _get_api_key(self) -> str | None:
        return self.config.api_key or os.getenv(self.config.api_key_env)

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is not None:
            return session
        session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        api_key = self._get_api_key()
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        for key, value in self.config.extra_headers.items():
            session.headers[key] = value
        self._session_local.session = session
        return session

    def _cache_key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _request_payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [msg.as_dict() if isinstance(msg, ChatMessage) else msg for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self.config.default_max_tokens,
        }
        payload.update(self.config.extra_payload)
        payload.update(request.extra_payload)
        return payload

    def _coerce_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        parts.append(stripped)
                    continue
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
            return "\n".join(parts).strip()
        return str(value).strip()

    def _extract_response_parts(self, response_json: dict[str, Any]) -> tuple[str, str | None]:
        choices = response_json.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0]
        if "message" in choice:
            message = choice.get("message") or {}
            reasoning = self._coerce_text(message.get("reasoning_content") or message.get("reasoning"))
            content = self._coerce_text(message.get("content"))
            if content:
                return content, (reasoning or None)
            return reasoning, (reasoning or None)
        return self._coerce_text(choice.get("text")), None

    def _is_retryable_exception(self, exc: requests.exceptions.RequestException) -> bool:
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            status_code = exc.response.status_code if exc.response is not None else None
            return status_code in self._RETRYABLE_HTTP_STATUS
        return False

    def _post_with_request_retries(self, session: requests.Session, payload: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
        request_attempt_limit = max(self.config.max_retries + 1, 1)
        last_exc: requests.exceptions.RequestException | None = None

        for request_attempt in range(request_attempt_limit):
            self._rate_limiter.wait()
            start = time.perf_counter()
            try:
                response = session.post(self._chat_url, json=payload, timeout=self.config.timeout_s)
                latency = time.perf_counter() - start
                response.raise_for_status()
                return response.json(), latency, request_attempt + 1
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                should_retry = self._is_retryable_exception(exc)
                if not should_retry or request_attempt + 1 >= request_attempt_limit:
                    raise

                backoff_s = min(2.0 ** request_attempt, 12.0)
                status_suffix = ""
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    status_suffix = f" status={exc.response.status_code}"
                print(
                    f"[ReasonBench] request_retry attempt={request_attempt + 1}/{request_attempt_limit} "
                    f"backoff_s={backoff_s:.1f}{status_suffix} error={exc!r}"
                )
                time.sleep(backoff_s)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("request_retry_exhausted_without_exception")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._request_payload(request)
        cache_key = self._cache_key(payload)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                text, raw = cached
                reasoning_content: str | None = None
                if isinstance(raw, dict):
                    _, reasoning_content = self._extract_response_parts(raw)
                return GenerationResult(text=text, raw_response=raw, from_cache=True, reasoning_content=reasoning_content)

        last_text = ""
        last_reasoning: str | None = None
        last_looping = False
        last_loop_reason = ""
        max_attempts = max(self.config.loop_retries, 1)
        session = self._get_session()
        total_request_attempts = 0
        for loop_attempt in range(max_attempts):
            raw_json, latency, request_attempts = self._post_with_request_retries(session, payload)
            total_request_attempts += request_attempts
            text, reasoning_content = self._extract_response_parts(raw_json)
            looping, reason = detect_loop(text, self._loop_config)
            last_text = text
            last_reasoning = reasoning_content
            last_looping = looping
            last_loop_reason = reason
            if looping and loop_attempt + 1 < max_attempts:
                time.sleep(1.5 * (loop_attempt + 1))
                continue
            metadata = {
                "loop_attempts": loop_attempt + 1,
                "request_attempts": total_request_attempts,
            }
            if looping:
                metadata.update({"loop_flag": True, "loop_reason": reason})
            result = GenerationResult(
                text=text,
                raw_response=raw_json,
                latency_s=latency,
                attempts=total_request_attempts,
                reasoning_content=reasoning_content,
                metadata=metadata,
            )
            if self.cache:
                self.cache.set(cache_key, text, raw_json)
            return result

        metadata = {}
        if last_looping:
            metadata = {"loop_flag": True, "loop_reason": last_loop_reason}
        return GenerationResult(
            text=last_text,
            attempts=total_request_attempts,
            reasoning_content=last_reasoning,
            metadata=metadata,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not self.supports_batch:
            return super().generate_batch(requests)
        prompts: list[str] = []
        for request in requests:
            if len(request.messages) == 1 and request.messages[0].role == "user":
                prompts.append(request.messages[0].content)
            else:
                flat = []
                for message in request.messages:
                    flat.append(f"[{message.role.upper()}]\n{message.content}")
                prompts.append("\n\n".join(flat))
        payload = {
            "model": self.config.model,
            "prompt": prompts,
            "temperature": requests[0].temperature if requests else self.config.default_temperature,
            "max_tokens": requests[0].max_tokens if requests and requests[0].max_tokens else self.config.default_max_tokens,
        }
        payload.update(self.config.extra_payload)
        self._rate_limiter.wait()
        session = self._get_session()
        start = time.perf_counter()
        response = session.post(self.config.completions_url, json=payload, timeout=self.config.timeout_s)
        latency = time.perf_counter() - start
        response.raise_for_status()
        raw_json = response.json()
        choices = raw_json.get("choices") or []
        choices.sort(key=lambda item: item.get("index", 0))
        results: list[GenerationResult] = []
        for index, _ in enumerate(prompts):
            text = ""
            reasoning_content: str | None = None
            if index < len(choices):
                choice = choices[index]
                if "message" in choice:
                    message = choice.get("message") or {}
                    reasoning = self._coerce_text(message.get("reasoning_content") or message.get("reasoning"))
                    content = self._coerce_text(message.get("content"))
                    text = content or reasoning
                    reasoning_content = reasoning or None
                else:
                    text = self._coerce_text(choice.get("text"))
            results.append(
                GenerationResult(
                    text=text,
                    raw_response=raw_json,
                    latency_s=latency / max(len(prompts), 1),
                    reasoning_content=reasoning_content,
                )
            )
        return results
