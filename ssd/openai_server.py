"""OpenAI-compatible HTTP server for SSD inference.

Supported endpoints:
- GET /health
- GET /v1/models
- GET /v1/models/{model_id}
- POST /v1/completions
- POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from ssd import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible inference server for SSD")

    # HTTP server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="info")

    # Model paths
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--draft", type=str, default=None, help="Draft model path (required when --spec)")
    parser.add_argument("--model-revision", type=str, default=None, help="HF revision for --model when using repo_id")
    parser.add_argument("--draft-revision", type=str, default=None, help="HF revision for --draft when using repo_id")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from HuggingFace Hub; only use local/cache files",
    )

    # SSD engine config
    parser.add_argument("--gpus", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--async", action="store_true", dest="async_spec", help="Enable async speculative decoding")
    parser.add_argument("--k", type=int, default=6, help="Speculative lookahead k")
    parser.add_argument("--f", type=int, default=3, help="Async speculative fan-out")
    parser.add_argument("--backup", type=str, choices=["jit", "fast"], default="jit", help="Spec backup strategy")
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graph path")
    parser.add_argument("--block-sz", type=int, default=256, help="KV cache block size")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Scheduler max_num_seqs")
    parser.add_argument("--x", type=float, default=None, help="Sampler-x coefficient")
    parser.add_argument("--verbose", action="store_true", help="Verbose engine logs")

    return parser.parse_args()


def _error(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": {"message": message, "type": "invalid_request_error"}})


def _as_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _normalize_text_prompt(prompt: Any) -> str | list[int]:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if not prompt:
            return ""
        if all(isinstance(t, int) for t in prompt):
            return prompt
        if len(prompt) == 1 and isinstance(prompt[0], str):
            return prompt[0]
        if len(prompt) == 1 and isinstance(prompt[0], list) and all(isinstance(t, int) for t in prompt[0]):
            return prompt[0]
    raise _error(
        400,
        "Only a single prompt is supported. Accepted prompt formats: string, token-id list, or single-item list.",
    )


def _to_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _to_finish_reason(token_count: int, max_new_tokens: int) -> str:
    if token_count >= max_new_tokens:
        return "length"
    return "stop"


def _parse_temperature_and_max_tokens(body: dict[str, Any]) -> tuple[float, int]:
    try:
        temperature = float(body.get("temperature", 1.0))
    except (TypeError, ValueError) as exc:
        raise _error(400, "temperature must be a number") from exc
    try:
        max_tokens = int(body.get("max_tokens", 16))
    except (TypeError, ValueError) as exc:
        raise _error(400, "max_tokens must be an integer") from exc

    if max_tokens <= 0:
        raise _error(400, "max_tokens must be > 0")
    if temperature < 0:
        raise _error(400, "temperature must be >= 0")
    return temperature, max_tokens


def _validate_local_model_dir(path: str, arg_name: str) -> str:
    cfg = os.path.join(path, "config.json")
    if not os.path.isdir(path):
        raise ValueError(f"{arg_name} must be an existing local directory: {path!r}")
    if not os.path.isfile(cfg):
        raise ValueError(f"{arg_name} must point to a model directory containing config.json: {path!r}")
    return path


def _resolve_model_ref(
    model_ref: str,
    arg_name: str,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    # 1) Prefer explicit local directory when it exists.
    local_candidate = os.path.abspath(os.path.expanduser(model_ref))
    if os.path.isdir(local_candidate):
        return _validate_local_model_dir(local_candidate, arg_name)

    # 2) If caller clearly passed a local path but it doesn't exist, fail early.
    if os.path.isabs(model_ref) or model_ref.startswith(".") or "\\" in model_ref:
        raise ValueError(f"{arg_name} local path does not exist: {model_ref!r}")

    # 3) Treat as HuggingFace repo id and resolve to snapshot directory.
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "huggingface_hub is required for repo-id loading. Install with `uv sync --extra server`."
        ) from exc

    try:
        snapshot_path = snapshot_download(
            repo_id=model_ref,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Failed to resolve {arg_name}={model_ref!r} from HuggingFace Hub: {exc}"
        ) from exc

    return _validate_local_model_dir(snapshot_path, arg_name)


class IncrementalDecoder:
    """Converts token id chunks into text deltas with special tokens removed."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prev_text = ""

    def push(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        self.ids.extend(token_ids)
        cur_text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        delta = cur_text[len(self.prev_text):]
        self.prev_text = cur_text
        return delta


@dataclass
class ServerState:
    llm: LLM
    model_name: str
    lock: threading.Lock

    def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
    ) -> tuple[dict[str, Any], int]:
        with self.lock:
            outputs, _ = self.llm.generate(
                [prompt],
                [sampling_params],
                use_tqdm=False,
                stream_callback=stream_callback,
            )
        output = outputs[0]
        token_ids = output["token_ids"]
        text = self.llm.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"text": text, "token_ids": token_ids}, len(token_ids)


def build_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="SSD OpenAI-Compatible API", version="0.1.0")

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": str(exc.detail)}})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        card = {"id": state.model_name, "object": "model", "created": now, "owned_by": "ssd"}
        return {"object": "list", "data": [card]}

    @app.get("/v1/models/{model_id:path}")
    async def retrieve_model(model_id: str) -> dict[str, Any]:
        if model_id != state.model_name:
            raise _error(404, f"Model '{model_id}' not found")
        now = int(time.time())
        return {"id": state.model_name, "object": "model", "created": now, "owned_by": "ssd"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise _error(400, "Request body must be a JSON object")

        prompt = _normalize_text_prompt(body.get("prompt"))
        temperature, max_tokens = _parse_temperature_and_max_tokens(body)
        ignore_eos = bool(body.get("ignore_eos", False))
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options")
        include_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage", False))

        req_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        sampling_params = SamplingParams(
            temperature=temperature,
            max_new_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )

        if isinstance(prompt, list):
            prompt_tokens = len(prompt)
        else:
            prompt_tokens = len(state.llm.tokenizer.encode(prompt, add_special_tokens=False))

        if not stream:
            output, completion_tokens = await asyncio.to_thread(state.generate, prompt, sampling_params, None)
            usage = _to_usage(prompt_tokens, completion_tokens)
            finish_reason = _to_finish_reason(completion_tokens, max_tokens)
            return {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": state.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": output["text"],
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(payload: dict[str, Any]) -> None:
            msg = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        def worker() -> None:
            decoder = IncrementalDecoder(state.llm.tokenizer)

            def on_tokens(_: int, new_ids: list[int]) -> None:
                delta = decoder.push(list(new_ids))
                if not delta:
                    return
                emit(
                    {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": state.model_name,
                        "choices": [{"index": 0, "text": delta, "logprobs": None, "finish_reason": None}],
                    }
                )

            try:
                _, completion_tokens = state.generate(prompt, sampling_params, on_tokens)
                finish_reason = _to_finish_reason(completion_tokens, max_tokens)
                emit(
                    {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": state.model_name,
                        "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": finish_reason}],
                    }
                )
                if include_usage:
                    emit(
                        {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": state.model_name,
                            "choices": [],
                            "usage": _to_usage(prompt_tokens, completion_tokens),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                emit({"error": {"message": f"Inference failed: {exc}", "type": "server_error"}})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, "data: [DONE]\n\n")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        async def event_stream():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise _error(400, "Request body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _error(400, "messages must be a non-empty list")

        normalized_messages: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise _error(400, "Each message must be an object")
            role = msg.get("role")
            if not isinstance(role, str):
                raise _error(400, "Each message must include role")
            normalized_messages.append({"role": role, "content": _as_text_content(msg.get("content", ""))})

        temperature, max_tokens = _parse_temperature_and_max_tokens(body)
        ignore_eos = bool(body.get("ignore_eos", False))
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options")
        include_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage", False))

        prompt_token_ids = state.llm.tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_tokens = len(prompt_token_ids)

        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        sampling_params = SamplingParams(
            temperature=temperature,
            max_new_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )

        if not stream:
            output, completion_tokens = await asyncio.to_thread(
                state.generate,
                prompt_token_ids,
                sampling_params,
                None,
            )
            usage = _to_usage(prompt_tokens, completion_tokens)
            finish_reason = _to_finish_reason(completion_tokens, max_tokens)
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": state.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output["text"]},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(payload: dict[str, Any]) -> None:
            msg = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        def worker() -> None:
            decoder = IncrementalDecoder(state.llm.tokenizer)

            # OpenAI-style first delta chunk announcing assistant role
            emit(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )

            def on_tokens(_: int, new_ids: list[int]) -> None:
                delta = decoder.push(list(new_ids))
                if not delta:
                    return
                emit(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.model_name,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                )

            try:
                _, completion_tokens = state.generate(prompt_token_ids, sampling_params, on_tokens)
                finish_reason = _to_finish_reason(completion_tokens, max_tokens)
                emit(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    }
                )
                if include_usage:
                    emit(
                        {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": state.model_name,
                            "choices": [],
                            "usage": _to_usage(prompt_tokens, completion_tokens),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                emit({"error": {"message": f"Inference failed: {exc}", "type": "server_error"}})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, "data: [DONE]\n\n")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        async def event_stream():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def create_state(args: argparse.Namespace) -> ServerState:
    if args.spec and not args.draft:
        raise ValueError("--draft is required when --spec is enabled")

    model_path = _resolve_model_ref(
        args.model,
        "--model",
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    draft_path = (
        _resolve_model_ref(
            args.draft,
            "--draft",
            revision=args.draft_revision,
            local_files_only=args.local_files_only,
        )
        if args.draft
        else model_path
    )

    llm_kwargs = dict(
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=args.async_spec,
        async_fan_out=args.f,
        draft=draft_path,
        kvcache_block_size=args.block_sz,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
        verbose=args.verbose,
    )
    llm = LLM(model_path, **llm_kwargs)
    return ServerState(llm=llm, model_name=args.model, lock=threading.Lock())


def main() -> None:
    args = parse_args()
    state = create_state(args)
    app = build_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
