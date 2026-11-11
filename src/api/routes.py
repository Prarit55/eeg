"""
FastAPI route handlers for query routing, EEG analysis, and knowledge graph exploration.
"""
from typing import Any, Dict, List

from src.api.models import EEGAnalysisRequest, QueryRequest, QueryResponse, SubQueryResponse
from src.eeg.pipeline import EEGPipeline
from src.knowledge.knowledge_graph import EpilepsyKnowledgeGraph
from src.knowledge.vector_store import MedicalVectorStore
from src.routing.router import QueryRouter
from src.self_rag.loop import SelfRAGLoop


def get_router():
    """Build and return an APIRouter instance if fastapi is installed."""
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError:
        raise ImportError("FastAPI is not installed. Run 'pip install fastapi uvicorn' to run the API.")

    router = APIRouter()

    # Lazy singletons
    _router_instance = QueryRouter()
    _eeg_instance = EEGPipeline()
    _graph_instance = EpilepsyKnowledgeGraph()
    _vector_instance = MedicalVectorStore()

    @router.get("/health")
    def health():
        return {"status": "ok", "service": "epilepsy-self-rag-api"}

    @router.post("/api/v1/query", response_model=QueryResponse)
    def query_endpoint(req: QueryRequest):
        loop = SelfRAGLoop(
            router=_router_instance,
            eeg_pipeline=_eeg_instance,
            vector_store=_vector_instance,
            knowledge_graph=_graph_instance,
        )
        res = loop.run(req.query, edf_path=req.edf_path, patient_history=req.patient_history)
        routed = res["routed"]

        sub_responses = []
        for r, sub_res in zip(routed.routes, res["sub_results"]):
            sub_responses.append(
                SubQueryResponse(
                    sub_query=r.sub_query,
                    activated_sources=r.activated,
                    answer=sub_res.answer,
                    citations=sub_res.citations,
                    confidence=sub_res.confidence,
                )
            )

        return QueryResponse(
            master_query=req.query,
            is_complex=routed.decomposition.is_complex,
            sub_queries=routed.decomposition.sub_queries,
            activated_modules=routed.all_activated(),
            results=sub_responses,
        )

    @router.post("/api/v1/eeg/analyze")
    def analyze_eeg(req: EEGAnalysisRequest):
        try:
            out = _eeg_instance.analyze(req.edf_path)
            probs = out["seizure_prob"]
            return {
                "edf_path": req.edf_path,
                "n_windows": int(out["n_windows"]),
                "mean_seizure_prob": float(probs.mean()),
                "max_seizure_prob": float(probs.max()),
                "rhythmic_spiking": bool(probs.max() > 0.5),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
