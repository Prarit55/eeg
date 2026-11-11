"""
FastAPI application entrypoint.
"""
def create_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI is not installed. Run 'pip install fastapi uvicorn' to run the API.")

    from src.api.routes import get_router

    app = FastAPI(
        title="Query-Adaptive Multimodal Self-RAG EEG Reasoning API",
        version="0.1.0",
        description="Backend API for ARP-455: Evidence-grounded reasoning over EEG, KG, and medical literature.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(get_router())
    return app
