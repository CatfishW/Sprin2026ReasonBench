from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from reasonbench.types import GenerationRequest, GenerationResult


class BaseLLMClient(ABC):
    @property
    def supports_batch(self) -> bool:
        return False

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [self.generate(request) for request in requests]
