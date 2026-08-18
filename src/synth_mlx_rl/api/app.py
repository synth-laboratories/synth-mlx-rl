"""FastAPI assembly.

Derived from the MIT-licensed `mlx-local-rl` prototype's app (see NOTICE), with
the two API families mounted as peers and the ``/v1/synth`` namespace added.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings
from ..protocols import LearnerEngine
from ..serialize import SingleThreadEngine
from ..schemas import (
    CheckpointRequest,
    CheckpointResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepRequest,
    OptimStepResponse,
    RenderChatRequest,
    RenderChatResponse,
    SampleRequest,
    SampleResponse,
    StateResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from ..snapshots import SnapshotError
from . import chat_completions, responses, synth
from .common import get_engine, idempotency_key, snapshot_http_error
from .idempotency import IdempotencyCache


def mounted_paths(app: Any) -> set[str]:
    """Every routable path, including inside nested/included routers.

    Recent FastAPI keeps an included router as one entry rather than flattening
    its routes, so a flat scan of ``app.routes`` silently finds nothing.
    """

    found: set[str] = set()
    stack = list(getattr(app, "routes", []))
    while stack:
        route = stack.pop()
        path = getattr(route, "path", None)
        if isinstance(path, str):
            found.add(path)
        nested = getattr(route, "routes", None)
        if nested:
            stack.extend(nested)
        original = getattr(route, "original_router", None)
        if original is not None:
            stack.extend(getattr(original, "routes", []))
    return found


def create_app(
    *,
    engine: LearnerEngine | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    configured_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # MLX streams are thread-local and FastAPI runs `def` endpoints in a
        # worker pool, so an engine built on this thread and called from there
        # raises `There is no Stream(gpu, N) in current thread` inside mx.eval.
        # The engine is therefore BUILT on the worker too, not merely called
        # from it; see serialize.py.
        if engine is not None:
            app.state.engine = SingleThreadEngine(engine)
        else:

            def build() -> Any:
                from ..engine import MLXEngine

                return MLXEngine(configured_settings)

            app.state.engine = SingleThreadEngine(factory=build)
        app.state.idempotency = IdempotencyCache()
        try:
            yield
        finally:
            app.state.engine.shutdown()

    app = FastAPI(
        title="synth-mlx-rl",
        version=__version__,
        description=(
            "A local Apple-Silicon MLX service for LoRA SFT and RLVR. Two "
            "OpenAI-compatible surfaces over one renderer and one rollout "
            "record. Start exactly one Uvicorn worker."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(SnapshotError)
    async def snapshot_error_handler(_: Request, exc: SnapshotError) -> JSONResponse:
        http = snapshot_http_error(exc)
        return JSONResponse(status_code=http.status_code, content=http.detail)

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        _: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "synth-mlx-rl", "docs": "/docs"}

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        state = get_engine(request).state()
        return {
            "ok": state.ready,
            "model": state.model,
            "training_version": state.training_version,
            "latest_policy_snapshot_id": state.latest_policy_snapshot_id,
        }

    @app.get("/v1/state", response_model=StateResponse)
    def state(request: Request) -> StateResponse:
        return get_engine(request).state()

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        current = get_engine(request).state()
        return {
            "object": "list",
            "data": [
                {"id": current.model, "object": "model", "owned_by": "synth-mlx-rl"}
            ],
        }

    @app.post("/v1/tokenize", response_model=TokenizeResponse)
    def tokenize(body: TokenizeRequest, request: Request) -> TokenizeResponse:
        return TokenizeResponse(
            token_ids=get_engine(request).encode(
                body.text, add_special_tokens=body.add_special_tokens
            )
        )

    @app.post("/v1/detokenize", response_model=DetokenizeResponse)
    def detokenize(body: DetokenizeRequest, request: Request) -> DetokenizeResponse:
        return DetokenizeResponse(
            text=get_engine(request).decode(
                body.token_ids, skip_special_tokens=body.skip_special_tokens
            )
        )

    @app.post("/v1/render_chat", response_model=RenderChatResponse)
    def render_chat(body: RenderChatRequest, request: Request) -> RenderChatResponse:
        return get_engine(request).render_chat(body)

    @app.post("/v1/sample", response_model=SampleResponse)
    def sample(body: SampleRequest, request: Request) -> SampleResponse:
        return get_engine(request).sample(
            body, idempotency_key=idempotency_key(request)
        )

    @app.post("/v1/forward_backward", response_model=ForwardBackwardResponse)
    def forward_backward(
        body: ForwardBackwardRequest, request: Request
    ) -> ForwardBackwardResponse:
        return get_engine(request).forward_backward(body)

    @app.post("/v1/optim_step", response_model=OptimStepResponse)
    def optim_step(body: OptimStepRequest, request: Request) -> OptimStepResponse:
        result = get_engine(request).optim_step(body.params)
        return OptimStepResponse.model_validate(result)

    @app.post("/v1/zero_grad")
    def zero_grad(request: Request) -> dict[str, int]:
        return {"cleared_accumulations": get_engine(request).zero_grad()}

    @app.post("/v1/checkpoints/save", response_model=CheckpointResponse)
    def save_checkpoint(
        body: CheckpointRequest, request: Request
    ) -> CheckpointResponse:
        return get_engine(request).save_checkpoint(body.name)

    @app.post("/v1/checkpoints/load", response_model=CheckpointResponse)
    def load_checkpoint(
        body: CheckpointRequest, request: Request
    ) -> CheckpointResponse:
        return get_engine(request).load_checkpoint(body.name)

    # Two families, mounted as peers. Neither is implemented in terms of the
    # other; both go through the same renderer and write the same record.
    app.include_router(chat_completions.router)
    app.include_router(responses.router)
    app.include_router(synth.router)

    missing = {"/v1/chat/completions", "/v1/responses"} - mounted_paths(app)
    if missing:  # pragma: no cover - assembly guard
        raise RuntimeError(
            "both API families must be mounted; missing " + ", ".join(sorted(missing))
        )

    return app
