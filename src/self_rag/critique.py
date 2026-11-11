"""
Self-RAG reflection critic: evaluates retrieved evidence and generation quality.
Reflection dimensions:
- IsRel: Evidence relevance to query
- IsSup: Answer grounding in evidence
- IsUse: Response utility and completeness
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from src.fusion.aggregator import EvidenceBundle


class ReflectionCritique(BaseModel):
    is_relevant: bool = Field(description="Is retrieved evidence relevant to the question?")
    is_supported: bool = Field(description="Is the reasoning directly supported by the evidence?")
    is_useful: bool = Field(description="Does the response satisfactorily answer the user's query?")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence (0.0 to 1.0).")
    critique_notes: str = Field(description="Brief explanation of the critique assessment.")
    needs_refinement: bool = Field(default=False, description="True if evidence is insufficient or contradictory.")


class SelfRAGCritic:
    """Evaluates evidence quality and grounding using rule-based heuristics or LLM."""

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold

    def critique(
        self,
        query: str,
        evidence: EvidenceBundle,
        generated_answer: Optional[str] = None,
    ) -> ReflectionCritique:
        """Evaluate evidence sufficiency and grounding."""
        has_ev = evidence.has_any_evidence()

        # Offline heuristic critique
        if not has_ev:
            return ReflectionCritique(
                is_relevant=False,
                is_supported=False,
                is_useful=False,
                confidence_score=0.20,
                critique_notes="No external evidence retrieved. Answer relies solely on prior parametric knowledge.",
                needs_refinement=True,
            )

        # Count evidence items
        n_docs = len(evidence.vector.chunks) if evidence.vector else 0
        n_kg = len(evidence.graph.triplets) if evidence.graph else 0
        has_eeg = evidence.eeg is not None

        # Compute heuristic score based on available evidence channels
        score = 0.50
        if n_docs > 0:
            score += 0.20
        if n_kg > 0:
            score += 0.15
        if has_eeg:
            score += 0.15
        score = min(0.95, score)

        is_supported = True
        if generated_answer:
            # Check for basic citation markers in generated text
            markers = ["[Doc-", "[KG-", "[EEG-", "[Patient-Context]"]
            has_citations = any(m in generated_answer for m in markers)
            if not has_citations and (n_docs + n_kg > 0):
                score -= 0.15

        is_useful = score >= self.confidence_threshold
        needs_refine = not is_useful

        return ReflectionCritique(
            is_relevant=True,
            is_supported=is_supported,
            is_useful=is_useful,
            confidence_score=round(score, 2),
            critique_notes=f"Retrieved {n_docs} doc chunk(s), {n_kg} KG triplet(s), EEG analyzed={has_eeg}.",
            needs_refinement=needs_refine,
        )
