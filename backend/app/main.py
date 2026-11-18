"""
FastAPI application entry point for Multimodal Research Co-pilot.
Configured with CORS, correlation ID middleware, static asset serving, and health checks.
"""

from __future__ import annotations
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router as api_v1_router
from backend.app.core.config import settings
from backend.app.core.logging import correlation_id_ctx, logger, setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(
        f"Gateway Base URL: {settings.GATEWAY_BASE_URL} (Mode: {settings.MODEL_MODE})"
    )
    logger.info(
        f"Embedding Engine: {settings.EMBEDDING_BACKEND} ({settings.EMBEDDING_DIMENSION}-d)"
    )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Grounded answers about research papers, with figure evidence and traceable citations.",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Assigns or propagates a correlation ID for request tracing (FR-OPS-04).
    """
    corr_id = request.headers.get("x-correlation-id") or f"corr_{uuid.uuid4().hex[:12]}"
    token = correlation_id_ctx.set(corr_id)
    try:
        response = await call_next(request)
        response.headers["x-correlation-id"] = corr_id
        return response
    finally:
        correlation_id_ctx.reset(token)


# Mount static assets for extracted figures and crops (FR-FIG-01, FR-ING-07)
crops_path = Path(settings.CROPS_DIR)
crops_path.mkdir(parents=True, exist_ok=True)
app.mount("/crops", StaticFiles(directory=str(crops_path)), name="crops")

# Mount web UI static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include versioned API routers
app.include_router(api_v1_router, prefix=settings.API_PREFIX)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "online",
    }
