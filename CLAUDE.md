# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
source venv/bin/activate
```

The venv uses Python 3.13 and already has dependencies installed (`fastapi`, `anthropic`, `uvicorn`).

## Running the server

```bash
uvicorn main:app --reload
```

## Architecture

This is an early-stage FastAPI project intended to become an agent/orchestration framework. Current structure:

- `main.py` — FastAPI entry point. Exposes `GET /` and `POST /chat` (streaming responses via Anthropic SDK using `claude-opus-4-8`).
- `api/` — intended for additional route modules
- `core/` — intended for shared logic
- `orchestrator/` — intended for agent orchestration logic
- `providers/` — intended for LLM/tool provider abstractions
- `capabilities/todo/` — capability definitions (currently empty)
- `skills/library/email_triage/` — skill definitions (currently empty)

The `POST /chat` endpoint accepts `{"messages": [{"role": "...", "content": "..."}]}` and streams plain text back via `StreamingResponse`.
