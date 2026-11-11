"""
Schemas for Query Understanding, Decomposition, and Adaptive Routing.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.routing.patient_context import PatientContext

MODULES: Dict[str, str] = {"s_vector": "Vector", "s_graph": "Graph", "s_eeg": "EEG"}


class QueryDecomposition(BaseModel):
    is_complex: bool = Field(description="True if the query contains several distinct information needs.")
    reasoning: str = Field(description="One or two sentences explaining the decision.")
    sub_queries: List[str] = Field(
        description="Self-contained atomic sub-queries. If the query is not complex, a list with the original query."
    )


class RoutingScores(BaseModel):
    s_vector: float = Field(ge=0.0, le=1.0, description="Need for medical literature / guideline text.")
    s_graph: float = Field(ge=0.0, le=1.0, description="Need for structured entity relations (knowledge graph).")
    s_eeg: float = Field(ge=0.0, le=1.0, description="Need for the patient's own EEG signal evidence.")


class SubQueryRoute(BaseModel):
    sub_query: str
    scores: RoutingScores
    activated: List[str] = Field(description="Subset of ['Vector', 'Graph', 'EEG'] with score >= tau.")
    retrieval_query: str = Field(
        default="", description="Query used for the vector search: sub_query enriched with patient history (if any)."
    )


class RoutedQuery(BaseModel):
    master_query: str
    decomposition: QueryDecomposition
    routes: List[SubQueryRoute]
    patient_context: Optional[PatientContext] = None

    def all_activated(self) -> List[str]:
        """Union of modules activated across sub-queries (in Vector, Graph, EEG order)."""
        used = {m for r in self.routes for m in r.activated}
        return [m for m in MODULES.values() if m in used]
