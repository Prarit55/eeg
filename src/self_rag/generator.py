"""
Evidence-Grounded Generator with citation tracking and uncertainty estimation.
Supports local reasoning synthesis and optional LLM backend integration.
"""
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from src.fusion.aggregator import EvidenceBundle
from src.fusion.context_formatter import format_multimodal_context
from src.self_rag.critique import ReflectionCritique, SelfRAGCritic


class EvidenceGroundedResponse(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    critique: ReflectionCritique
    disclaimer: str = (
        "Disclaimer: This system is designed for medical research evaluation and decision support, "
        "not for autonomous clinical diagnosis or treatment."
    )


class EvidenceGroundedGenerator:
    """Produces structured, evidence-grounded medical answers with citation tags."""

    def __init__(self, critic: Optional[SelfRAGCritic] = None, llm: Optional[Any] = None) -> None:
        self.critic = critic or SelfRAGCritic()
        self.llm = llm

    def generate(self, bundle: EvidenceBundle) -> EvidenceGroundedResponse:
        context_str = format_multimodal_context(bundle)
        citations: List[str] = []
        answer_parts: List[str] = []

        # 1. Patient Context Grounding
        if bundle.patient_context and not bundle.patient_context.is_empty():
            ctx_summary = bundle.patient_context.summary()
            flags = bundle.patient_context.eeg_flags()
            flag_text = f" (Clinical note: {'; '.join(flags)})" if flags else ""
            answer_parts.append(f"Patient Profile: {ctx_summary}{flag_text} [Patient-Context].")
            citations.append("Patient-Context")

        # 2. Scalp EEG Signal Evidence
        if bundle.eeg:
            answer_parts.append(
                f"Signal Analysis: Scalp EEG evaluation shows {bundle.eeg.summary} [EEG-Evidence]."
            )
            citations.append("EEG-Evidence")

        # 3. Knowledge Graph Structured Relationships
        if bundle.graph and bundle.graph.triplets:
            kg_summary = "; ".join(bundle.graph.triplets[:4])
            answer_parts.append(
                f"Knowledge Graph Relationships: {kg_summary} [KG-Triplets]."
            )
            citations.append("KG-Triplets")

        # 4. Clinical Guidelines & Medical Literature
        if bundle.vector and bundle.vector.chunks:
            for i, chunk in enumerate(bundle.vector.chunks[:2], 1):
                clean_chunk = chunk.strip().replace("\n", " ")
                # Truncate at sentence boundary if possible
                if len(clean_chunk) > 280:
                    clean_chunk = clean_chunk[:280] + "..."
                answer_parts.append(
                    f"Clinical Guidelines: '{clean_chunk}' [Doc-{i}]."
                )
                citations.append(f"Doc-{i}")

        if not answer_parts:
            answer_parts.append(
                "No external evidence retrieved for this query. "
                "Consult standardized epilepsy clinical protocols and specialist guidance."
            )

        final_answer = "\n\n".join(answer_parts)
        critique = self.critic.critique(bundle.query, bundle, final_answer)

        return EvidenceGroundedResponse(
            query=bundle.query,
            answer=final_answer,
            citations=citations,
            confidence=critique.confidence_score,
            critique=critique,
        )
