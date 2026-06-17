import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from providers import get_provider

router = APIRouter()
provider = get_provider()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-8")
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS


@router.get("/models")
def list_models():
    return {"models": provider.list_models()}


@router.post("/chat")
def chat(request: ChatRequest):
    def stream():
        yield from provider.stream(
            messages=[m.model_dump() for m in request.messages],
            model=request.model,
            max_tokens=request.max_tokens,
        )

    return StreamingResponse(stream(), media_type="text/plain")
