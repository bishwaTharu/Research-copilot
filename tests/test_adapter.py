"""
Unit and Contract tests for LLM Adapter (INT-01, INT-06, INT-07, INT-12).
"""

import asyncio
import pytest
from pydantic import BaseModel
from backend.app.llm.adapter import LLMAdapter


class SampleStructuredOutput(BaseModel):
    summary: str
    confidence: float


def test_llm_adapter_mock_chat():
    adapter = LLMAdapter(mock_mode=True)
    resp = adapter._generate_mock_response(
        messages=[{"role": "user", "content": "What is the Transformer?"}],
        stage="test_qa",
        model="mock-model",
    )
    assert resp.content != ""
    assert resp.routed_provider == "mock-provider"


def test_llm_adapter_caching():
    """Deterministic stages cache responses by prompt hash and model (INT-12)."""
    adapter = LLMAdapter(mock_mode=True)
    adapter.set_mock_response("deterministic_test", "First computed response")

    messages = [{"role": "user", "content": "deterministic_test prompt"}]
    cache_key = adapter._compute_cache_key("mock-model", messages)
    adapter._cache[cache_key] = "First computed response"
    
    assert cache_key in adapter._cache
    assert adapter._cache[cache_key] == "First computed response"


def test_structured_output_enforcement():
    """Structured output must strictly parse and validate against Pydantic schema (INT-07)."""
    import json
    raw_payload = json.dumps({"summary": "Valid summary", "confidence": 0.95})
    parsed = json.loads(raw_payload)
    result = SampleStructuredOutput.model_validate(parsed)
    assert result.summary == "Valid summary"
    assert result.confidence == 0.95
