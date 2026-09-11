"""
MiniVal OpenAI-Compatible API Server
=====================================
Server API FastAPI yang kompatibel penuh dengan format OpenAI (/v1/chat/completions).
Mendukung:
  - Streaming Server-Sent Events (SSE)
  - Non-streaming response
  - Endpoint model metadata (/v1/models)
  - Health check (/health)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from queue import Queue
from threading import Thread
from typing import Annotated, AsyncGenerator, List, Literal, Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# State & Resource Management

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "minival-v1"))

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        print(f" Memuat model dari: {MODEL_PATH} ({DEVICE})...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        raw_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if DEVICE == "cuda":
            _model = raw_model.half().eval().to(DEVICE)
        else:
            _model = raw_model.float().eval().to(DEVICE)
        print(" Model siap melayani permintaan API!")
    return _model, _tokenizer


# Pydantic Schemas (OpenAI-compatible)

security_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security_scheme)]
) -> Optional[str]:
    required_key = os.getenv("MINIVAL_API_KEY")
    if not required_key:
        return None
    if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = Field(
        ..., description="Role pengirim ('user', 'assistant', 'system', 'tool')"
    )
    content: str = Field(..., min_length=1, description="Isi teks pesan")


class ChatRequest(BaseModel):
    model: str = Field(default="minival-v1", description="Nama model")
    messages: List[ChatMessage] = Field(..., min_items=1, description="Riwayat percakapan")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.90, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = Field(default=True, description="Aktifkan streaming SSE")


class QueueStreamer(TextStreamer):
    """Bridge antara token generator transformers dan SSE response stream."""

    def __init__(self, tokenizer, queue: Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


# Factory Pattern: create_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_and_tokenizer()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MiniVal API",
        version="1.1.0",
        description="OpenAI-compatible inference API for MiniVal LLM",
        lifespan=lifespan,
    )

    cors_origins_env = os.getenv("CORS_ORIGINS", "")
    if cors_origins_env.strip():
        origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    else:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "device": DEVICE, "model_loaded": _model is not None}

    @app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "minival-v1",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "valincia",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
    async def chat_completions(req: ChatRequest):
        model, tokenizer = get_model_and_tokenizer()
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model belum siap dimuat")

        messages_data = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(messages_data, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        if req.stream:
            stream_queue: Queue = Queue()
            streamer = QueueStreamer(tokenizer, stream_queue)

            def _worker():
                with torch.no_grad():
                    model.generate(
                        inputs.input_ids,
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        do_sample=True,
                        streamer=streamer,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

            Thread(target=_worker, daemon=True).start()

            async def event_generator() -> AsyncGenerator[str, None]:
                created = int(time.time())
                while True:
                    await asyncio.sleep(0.005)
                    while not stream_queue.empty():
                        token = stream_queue.get_nowait()
                        if token is None:
                            yield "data: [DONE]\n\n"
                            return
                        chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        def _generate():
            with torch.no_grad():
                return model.generate(
                    inputs.input_ids,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

        outputs = await asyncio.to_thread(_generate)

        gen_tokens = outputs[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        prompt_len = len(inputs.input_ids[0])
        gen_len = len(gen_tokens)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_len,
                "total_tokens": prompt_len + gen_len,
            },
        }

    return app


app = create_app()


def run(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniVal API Server")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="Host address")
    parser.add_argument("--port", default=8000, type=int, help="Port number")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
