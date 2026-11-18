"""
LLM Adapter for FreeLLMAPI and OpenAI-compatible endpoints.
Implements INT-01 to INT-08, INT-12, and ADR-5.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar
import aiohttp
from pydantic import BaseModel, ValidationError

from backend.app.core.config import settings
from backend.app.core.logging import logger, log_llm_call

T = TypeVar("T", bound=BaseModel)


class RateLimitException(Exception):
    """Raised when FreeLLMAPI or upstream returns 429 quota exhaustion (INT-06)."""

    pass


class GatewayUnavailableException(Exception):
    """Raised when the gateway or all upstream providers are unreachable (INT-10)."""

    pass


class ModelSwitchDetectedWarning(UserWarning):
    """Raised/logged when a different provider/model answers inside the same task (INT-04)."""

    pass


class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    routed_provider: Optional[str] = None
    routed_model: Optional[str] = None
    cache_hit: bool = False


class LLMAdapter:
    """
    Unified entry point to FreeLLMAPI and OpenAI-compatible endpoints.
    Provides:
      - Exponential backoff with jitter (INT-06)
      - Header inspection for routed provider/model (INT-03)
      - Pinned vs Auto model routing (INT-04)
      - Deterministic stage prompt-hash caching (INT-12)
      - Structured output validation with bounded retries (INT-07)
      - Mock mode for offline development & CI
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.base_url = (base_url or settings.GATEWAY_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.GATEWAY_API_KEY
        self.mock_mode = mock_mode
        self._cache: Dict[str, str] = {}
        self._mock_responses: Dict[str, Any] = {}

    def _compute_cache_key(self, model: str, messages: List[Dict[str, str]]) -> str:
        serialized = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        use_cache: bool = False,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Executes a chat completion call with retries and header tracking.
        """
        # Select model mode (INT-04)
        if model is None:
            model = (
                "auto"
                if settings.MODEL_MODE == "auto"
                else settings.PINNED_DEFAULT_MODEL
            )

        # Mock mode fallback (instant offline execution)
        if self.mock_mode:
            if use_cache:
                cache_key = self._compute_cache_key(model, messages)
                if cache_key in self._cache:
                    return LLMResponse(
                        content=self._cache[cache_key],
                        routed_provider="local-cache",
                        routed_model=model,
                        cache_hit=True,
                    )
                resp = self._generate_mock_response(messages, stage, model)
                self._cache[cache_key] = resp.content
                return resp
            return self._generate_mock_response(messages, stage, model)

        # Cache lookup for deterministic stages (INT-12)
        cache_key = self._compute_cache_key(model, messages)
        if use_cache and cache_key in self._cache:
            cached_content = self._cache[cache_key]
            log_llm_call(
                stage=stage,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                status="success (cache)",
                routed_provider="local-cache",
                routed_model=model,
                cache_hit=True,
            )
            return LLMResponse(
                content=cached_content,
                routed_provider="local-cache",
                routed_model=model,
                cache_hit=True,
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        retries = 0
        backoff = settings.BASE_BACKOFF_SECONDS

        while retries <= settings.MAX_RETRIES:
            start_time = time.perf_counter()
            try:
                timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT_SECONDS)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        latency_ms = (time.perf_counter() - start_time) * 1000

                        # Capture routed provider and model from headers (INT-03)
                        routed_provider = (
                            resp.headers.get("x-provider")
                            or resp.headers.get("x-routed-provider")
                            or "freellmapi"
                        )
                        routed_model = (
                            resp.headers.get("x-model")
                            or resp.headers.get("x-routed-model")
                            or model
                        )

                        if resp.status == 200:
                            data = await resp.json()
                            content = data["choices"][0]["message"]["content"]
                            usage = data.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)

                            log_llm_call(
                                stage=stage,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                latency_ms=latency_ms,
                                status="success",
                                routed_provider=routed_provider,
                                routed_model=routed_model,
                            )

                            if use_cache:
                                self._cache[cache_key] = content

                            return LLMResponse(
                                content=content,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens,
                                latency_ms=latency_ms,
                                routed_provider=routed_provider,
                                routed_model=routed_model,
                            )

                        elif resp.status in (429, 502, 503, 504):
                            # Backoff with full jitter (INT-06)
                            jitter = random.uniform(0.5, 1.5)
                            sleep_time = min(
                                backoff * jitter, settings.MAX_BACKOFF_SECONDS
                            )
                            logger.warning(
                                f"Gateway returned {resp.status} on stage '{stage}'. "
                                f"Retrying in {sleep_time:.2f}s (attempt {retries + 1}/{settings.MAX_RETRIES})."
                            )
                            await asyncio.sleep(sleep_time)
                            backoff *= 2
                            retries += 1
                        else:
                            error_text = await resp.text()
                            log_llm_call(
                                stage=stage,
                                prompt_tokens=0,
                                completion_tokens=0,
                                latency_ms=latency_ms,
                                status=f"error_{resp.status}",
                                routed_provider=routed_provider,
                                routed_model=routed_model,
                                error=error_text[:200],
                            )
                            raise GatewayUnavailableException(
                                f"Gateway returned HTTP {resp.status}: {error_text[:200]}"
                            )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.warning(
                    f"Connection error to gateway on stage '{stage}': {exc}. Retrying..."
                )
                retries += 1
                if retries > settings.MAX_RETRIES:
                    log_llm_call(
                        stage=stage,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        status="failure_network",
                        error=str(exc),
                    )
                    raise GatewayUnavailableException(
                        f"Failed to connect to gateway at {url}: {exc}"
                    ) from exc
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RateLimitException(
            f"Exhausted {settings.MAX_RETRIES} retries due to quota limits or gateway errors."
        )

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """
        Streams partial tokens for responsive UI updates (FR-QA-06).
        """
        if self.mock_mode:
            mock_resp = self._generate_mock_response(messages, stage, model)
            for word in mock_resp.content.split(" "):
                yield word + " "
                await asyncio.sleep(0.02)
            return

        if model is None:
            model = (
                "auto"
                if settings.MODEL_MODE == "auto"
                else settings.PINNED_DEFAULT_MODEL
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise GatewayUnavailableException(
                        f"Stream request failed: {resp.status} - {text}"
                    )

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_part = line_str[6:]
                        if data_part == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_part)
                            delta = parsed["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue

    async def structured_completion(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        stage: str,
        model: Optional[str] = None,
        max_validation_retries: int = 2,
    ) -> T:
        """
        Enforces structured output validation against a Pydantic schema (INT-07).
        Appends schema instructions and re-prompts on JSON / schema validation failures.
        """
        system_instruction = (
            f"\n\nYou MUST respond ONLY with a valid JSON object matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            f"Do not include any markdown backticks, prefix, or explanation. Output raw JSON only."
        )

        current_messages = list(messages)
        # Append instruction to last message or system prompt
        if current_messages and current_messages[0]["role"] == "system":
            current_messages[0] = {
                "role": "system",
                "content": current_messages[0]["content"] + system_instruction,
            }
        else:
            current_messages.insert(
                0, {"role": "system", "content": system_instruction}
            )

        for attempt in range(max_validation_retries + 1):
            response = await self.chat_completion(
                messages=current_messages,
                stage=stage,
                model=model,
                temperature=0.0,
            )
            raw_text = response.content.strip()
            # Clean possible markdown fencing
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            try:
                parsed_json = json.loads(raw_text)
                validated_obj = schema.model_validate(parsed_json)
                return validated_obj
            except (json.JSONDecodeError, ValidationError) as err:
                logger.warning(
                    f"Structured validation failed on attempt {attempt + 1}/{max_validation_retries + 1} "
                    f"for stage '{stage}': {err}"
                )
                if attempt < max_validation_retries:
                    # Feed back error to model for correction
                    current_messages.append({"role": "assistant", "content": raw_text})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": f"Your previous response had validation error: {err}. Please output strictly valid JSON matching the schema.",
                        }
                    )
                else:
                    raise ValueError(
                        f"Failed to produce structured output matching schema {schema.__name__}: {err}"
                    )

        raise ValueError("Structured completion failed.")

    def set_mock_response(self, stage_or_keyword: str, response: Any):
        """Allows test suites to configure mock responses."""
        self._mock_responses[stage_or_keyword] = response

    def _generate_mock_response(
        self, messages: List[Dict[str, str]], stage: str, model: Optional[str]
    ) -> LLMResponse:
        """Deterministic mock response for offline CI and testing."""
        # Check custom mock registry
        for key, val in self._mock_responses.items():
            if key in stage or any(key in m.get("content", "") for m in messages):
                content = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
                return LLMResponse(
                    content=content,
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    latency_ms=15.0,
                    routed_provider="mock-provider",
                    routed_model=model or "mock-model",
                )

        # Default fallback mock response
        if "orchestrator" in stage:
            default_dict = {
                "has_sufficient_evidence": True,
                "abstention_reason": None,
                "summary": "This is a grounded mock synthesis explaining the paper findings based on the retrieved evidence.",
                "claims": [
                    {
                        "claim_id": "c1",
                        "claim": "The system verifies claims against retrieved blocks and figures.",
                        "evidence_items": [
                            {"type": "text", "id": "b1", "quote": "Streaming answers are verified"}
                        ],
                    }
                ],
                "suggested_followups": [
                    "What empirical evaluations are reported in this work?",
                    "What are the primary acknowledged failure modes?",
                ],
            }
            return LLMResponse(
                content=json.dumps(default_dict),
                prompt_tokens=15,
                completion_tokens=40,
                total_tokens=55,
                latency_ms=10.0,
                routed_provider="mock-provider",
                routed_model=model or "mock-model",
            )

        if "verifier" in stage or "entailment" in stage:
            verdict_dict = {
                "verdict": "supported",
                "reasoning": "Every assertion is directly entailed by the provided evidence text.",
                "supported_points": ["Point supported"],
                "unsupported_points": [],
            }
            return LLMResponse(
                content=json.dumps(verdict_dict),
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
                latency_ms=8.0,
                routed_provider="mock-provider",
                routed_model=model or "mock-model",
            )

        default_content = "This is a deterministic mock response from the research co-pilot LLM adapter."
        return LLMResponse(
            content=default_content,
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
            latency_ms=10.0,
            routed_provider="mock-provider",
            routed_model=model or "mock-model",
        )


# Global adapter instance
llm_adapter = LLMAdapter()
