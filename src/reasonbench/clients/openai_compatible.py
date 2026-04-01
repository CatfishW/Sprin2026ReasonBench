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
    def __init__(self, config: ClientConfig, cache: SQLiteCache | None = None):
        self.config = config
        self.cache = cache
        self._rate_limiter = MinIntervalRateLimiter(config.min_request_interval_s)
        self._session_local = threading.local()
        self._loop_config = LoopDetectionConfig(length_ceiling=config.loop_length_ceiling)

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

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if "message" in choice:
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content") or ""
            content = message.get("content") or ""
            if reasoning and content:
                return f"{reasoning}\n\n{content}".strip()
            return (content or reasoning or "").strip()
        return str(choice.get("text") or "").strip()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._request_payload(request)
        cache_key = self._cache_key(payload)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                text, raw = cached
                return GenerationResult(text=text, raw_response=raw, from_cache=True)

        last_text = ""
        session = self._get_session()
        for loop_attempt in range(max(self.config.loop_retries, 1)):
            self._rate_limiter.wait()
            start = time.perf_counter()
            response = session.post(self.config.base_url, json=payload, timeout=self.config.timeout_s)
            latency = time.perf_counter() - start
            response.raise_for_status()
            raw_json = response.json()
            text = self._extract_text(raw_json)
            looping, reason = detect_loop(text, self._loop_config)
            last_text = text
            if looping and loop_attempt + 1 < max(self.config.loop_retries, 1):
                time.sleep(1.5 * (loop_attempt + 1))
                continue
            result = GenerationResult(
                text=text,
                raw_response=raw_json,
                latency_s=latency,
                attempts=loop_attempt + 1,
                metadata={"loop_flag": looping, "loop_reason": reason} if looping else {},
            )
            if self.cache:
                self.cache.set(cache_key, text, raw_json)
            return result

        return GenerationResult(text=last_text)

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
            if index < len(choices):
                text = str(choices[index].get("text") or "").strip()
            results.append(GenerationResult(text=text, raw_response=raw_json, latency_s=latency / max(len(prompts), 1)))
        return results
