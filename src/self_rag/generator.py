"""
Evidence-Grounded Generator with citation tracking and uncertainty estimation.
"""
from typing import List, Optional
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

    def __init__(self, critic: Optional[SelfRAGCritic] = None) -> None:
        self.critic = critic or SelfRAGCritic()

    def generate(self, bundle: EvidenceBundle) -> EvidenceGroundedResponse:
        context_str = format_multimodal_context(bundle)
        citations: List[str] = []

        # Build citation list and synthesized answer
        answer_parts: List[str] = []

        if bundle.eeg:
            answer_parts.append(
                f"Signal Analysis: EEG analysis reveals {bundle.eeg.summary} [EEG-Evidence]."
            )
            citations.append("EEG-Evidence")

        if bundle.graph and bundle.graph.triplets:
            kg_summary = "; ".join(bundle.graph.triplets[:3])
            answer_parts.append(
                f"Knowledge Graph Relationships: {kg_summary} [KG-Triplets]."
            )
            citations.append("KG-Triplets")

        if bundle.vector and bundle.vector.chunks:
            doc_sample = bundle.vector.chunks[0][:150].replace("\n", " ")
            answer_parts.append(
                f"Clinical Guidelines: Literature indicates '{doc_sample}...' [Doc-1]."
            )
            citations.append("Doc-1")

        if not answer_parts:
            answer_parts.append(
                "Unable to retrieve external evidence for this query. "
                "Consult standardized epilepsy clinical protocols."
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
