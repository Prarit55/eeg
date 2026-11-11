"""
Pydantic API request and response models.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The clinical or master question.")
    patient_history: str = Field(default="", description="Optional patient clinical history string.")
    edf_path: Optional[str] = Field(default=None, description="Optional path to EEG .edf recording.")
    tau: float = Field(default=0.5, ge=0.0, le=1.0, description="Routing activation threshold.")


class EEGAnalysisRequest(BaseModel):
    edf_path: str = Field(..., description="Path to CHB-MIT .edf file to analyze.")


class SubQueryResponse(BaseModel):
    sub_query: str
    activated_sources: List[str]
    answer: str
    citations: List[str]
    confidence: float


class QueryResponse(BaseModel):
    master_query: str
    is_complex: bool
    sub_queries: List[str]
    activated_modules: List[str]
    results: List[SubQueryResponse]
