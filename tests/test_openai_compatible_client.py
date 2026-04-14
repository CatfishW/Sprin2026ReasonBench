from reasonbench.clients.openai_compatible import OpenAICompatibleClient
from reasonbench.config import ClientConfig
import requests


def _client() -> OpenAICompatibleClient:
    config = ClientConfig(base_url="http://localhost:9999/v1/chat/completions", model="dummy")
    return OpenAICompatibleClient(config)


def test_extract_response_parts_prefers_content_and_keeps_reasoning() -> None:
    client = _client()
    text, reasoning = client._extract_response_parts(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "internal reasoning",
                        "content": "final answer",
                    }
                }
            ]
        }
    )
    assert text == "final answer"
    assert reasoning == "internal reasoning"


def test_extract_response_parts_falls_back_to_reasoning_if_content_missing() -> None:
    client = _client()
    text, reasoning = client._extract_response_parts(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "reason-only",
                        "content": "",
                    }
                }
            ]
        }
    )
    assert text == "reason-only"
    assert reasoning == "reason-only"


def test_extract_response_parts_text_completion() -> None:
    client = _client()
    text, reasoning = client._extract_response_parts({"choices": [{"text": "plain completion"}]})
    assert text == "plain completion"
    assert reasoning is None


def test_retryable_exception_timeout() -> None:
    client = _client()
    exc = requests.exceptions.ReadTimeout("timed out")
    assert client._is_retryable_exception(exc) is True


def test_retryable_exception_http_500() -> None:
    client = _client()
    response = requests.Response()
    response.status_code = 500
    exc = requests.exceptions.HTTPError(response=response)
    assert client._is_retryable_exception(exc) is True


def test_non_retryable_exception_http_400() -> None:
    client = _client()
    response = requests.Response()
    response.status_code = 400
    exc = requests.exceptions.HTTPError(response=response)
    assert client._is_retryable_exception(exc) is False
