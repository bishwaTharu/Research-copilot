# ADR-5: Adapter Interface Around Language-Model Gateway

## Status
Accepted

## Context
FreeLLMAPI routes requests across free-tier LLM providers. Provider availability, daily token quotas, and terms of service fluctuate frequently. Furthermore, tests and CI environments must run deterministically without burning rate limits or relying on live external networks.

## Decision
All language model interactions must pass through an `LLMAdapter` abstraction (`INT-01`, `INT-02`):
1. **OpenAI Compatibility**: Communicates over standard OpenAI-compatible `/v1/chat/completions` REST endpoints.
2. **Metadata Extraction**: Inspects response headers (`x-provider`, `x-model`) to audit provider routing, token count, latency, and pipeline stage (`INT-03`).
3. **Resilience & Backoff**: Intercepts HTTP 429 and 5xx responses with exponential backoff and jitter (`INT-06`).
4. **Structured Output Enforcer**: Validates model output against Pydantic / JSON schema with bounded retry attempts (`INT-07`).
5. **Deterministic Ingestion Caching**: In deterministic pipeline stages, responses are cached keyed on `sha256(prompt + model)` (`INT-12`).
6. **Pluggable Implementation**: Supports switching from `FreeLLMAPI` to a local mock server (`MockLLMGateway`), direct OpenAI, Ollama, or vLLM by changing environment configuration only.

## Consequences
- **Pros**: Complete separation of business logic from provider idiosyncrasies. Repeatable test execution in CI using recorded fixtures or mock gateway.
- **Cons**: Minor abstraction overhead; necessitates comprehensive mock testing fixtures.

