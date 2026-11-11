"""
Evidence aggregation: bundles heterogeneous multimodal evidence (EEG, KG, literature, history).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.routing.patient_context import PatientContext


class EEGEvidence(BaseModel):
    recording_id: str = "edf_recording"
    n_windows: int
    mean_seizure_prob: float
    max_seizure_prob: float
    rhythmic_spiking_detected: bool
    summary: str


class TripletEvidence(BaseModel):
    triplets: List[str] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)


class DocumentEvidence(BaseModel):
    chunks: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """Unified container storing all retrieved and processed evidence for a query."""
    query: str
    patient_context: Optional[PatientContext] = None
    eeg: Optional[EEGEvidence] = None
    graph: Optional[TripletEvidence] = None
    vector: Optional[DocumentEvidence] = None

    def has_any_evidence(self) -> bool:
        has_eeg = self.eeg is not None
        has_kg = bool(self.graph and self.graph.triplets)
        has_doc = bool(self.vector and self.vector.chunks)
        has_ctx = bool(self.patient_context and not self.patient_context.is_empty())
        return has_eeg or has_kg or has_doc or has_ctx


class EvidenceAggregator:
    """Aggregates retrieved outputs from EEG analysis, graph queries, and vector searches."""

    def __init__(self) -> None:
        pass

    def build_bundle(
        self,
        query: str,
        patient_context: Optional[PatientContext] = None,
        eeg_analysis: Optional[Dict[str, Any]] = None,
        graph_triplets: Optional[List[str]] = None,
        seed_entities: Optional[List[str]] = None,
        vector_chunks: Optional[List[str]] = None,
        doc_sources: Optional[List[str]] = None,
    ) -> EvidenceBundle:
        eeg_ev = None
        if eeg_analysis is not None:
            probs = eeg_analysis.get("seizure_prob")
            mean_p = float(probs.mean()) if probs is not None else 0.0
            max_p = float(probs.max()) if probs is not None else 0.0
            rhythmic = bool(max_p > 0.5)
            summary = (
                f"Analyzed {int(eeg_analysis.get('n_windows', 0))} windows. "
                f"Mean seizure prob: {mean_p:.2f}, peak seizure prob: {max_p:.2f}. "
                f"Rhythmic spiking/seizure activity: {'Detected' if rhythmic else 'None'}."
            )
            eeg_ev = EEGEvidence(
                n_windows=int(eeg_analysis.get("n_windows", 0)),
                mean_seizure_prob=mean_p,
                max_seizure_prob=max_p,
                rhythmic_spiking_detected=rhythmic,
                summary=summary,
            )

        graph_ev = None
        if graph_triplets:
            graph_ev = TripletEvidence(
                triplets=graph_triplets,
                seed_entities=seed_entities or [],
            )

        vector_ev = None
        if vector_chunks:
            vector_ev = DocumentEvidence(
                chunks=vector_chunks,
                sources=doc_sources or ["Clinical Guidelines"] * len(vector_chunks),
            )

        return EvidenceBundle(
            query=query,
            patient_context=patient_context,
            eeg=eeg_ev,
            graph=graph_ev,
            vector=vector_ev,
        )
