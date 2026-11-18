"""
Structured logging with correlation IDs and LLM call auditing.
Compliant with FR-OPS-04, INT-03, and INT-11.
"""

import sys
import json
import logging
import contextvars
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable for tracing a request across asynchronous stages (FR-OPS-04)
correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with timestamp, level, correlation_id, and context."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_ctx.get()
            or getattr(record, "correlation_id", "none"),
        }

        # Include extra payload if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_data.update(record.extra_fields)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures the root application logger with structured JSON output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())

    logger = logging.getLogger("copilot")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()


def log_llm_call(
    stage: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    status: str,
    routed_provider: Optional[str] = None,
    routed_model: Optional[str] = None,
    cache_hit: bool = False,
    error: Optional[str] = None,
):
    """
    Persists and logs LLM call details (INT-03, FR-OPS-02).
    Tracks routed provider, routed model from FreeLLMAPI headers, token usage, latency.
    """
    extra = {
        "event": "llm_call",
        "stage": stage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "routed_provider": routed_provider or "unknown",
        "routed_model": routed_model or "unknown",
        "cache_hit": cache_hit,
    }
    if error:
        extra["error"] = error

    logger.info(
        f"LLM call [{stage}]: {status} via {routed_provider}/{routed_model} in {latency_ms:.1f}ms",
        extra={"extra_fields": extra},
    )
