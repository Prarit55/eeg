"""
Query Intent Classifier.
Classifies clinical questions into synopsis-defined categories:
- definitional
- relational
- eeg_specific
- clinical
- complex
"""
import enum
import re
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryIntent(str, enum.Enum):
    DEFINITIONAL = "definitional"
    RELATIONAL = "relational"
    EEG_SPECIFIC = "eeg_specific"
    CLINICAL = "clinical"
    COMPLEX = "complex"


class QueryClassification(BaseModel):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


_PATIENT_SIGNAL = re.compile(
    r"\bthis patient|\bthe patient|\bmy patient|\bpatient'?s\b|"
    r"\b(this|the|my|his|her|their)\s+(eeg|recording|signal|trace|scan|segment|window)\b", re.I
)
_RELATIONAL = re.compile(r"region|lobe|cortex|associated|affect|related|linked|comorbid|part of|type of", re.I)
_DEFINITIONAL = re.compile(r"what is|what are|define|definition|explain", re.I)
_CLINICAL = re.compile(r"guideline|criteria|treat|first-line|dose|medication|drug|therapy|management", re.I)


def classify_query(query: str, is_complex: bool = False) -> QueryClassification:
    """Heuristic rule-based query intent classification."""
    if is_complex:
        return QueryClassification(
            intent=QueryIntent.COMPLEX,
            confidence=0.95,
            explanation="Query contains multiple sub-questions or disparate information needs.",
        )
    if _PATIENT_SIGNAL.search(query):
        return QueryClassification(
            intent=QueryIntent.EEG_SPECIFIC,
            confidence=0.90,
            explanation="Query references patient-specific EEG recordings or waveform properties.",
        )
    if _DEFINITIONAL.search(query):
        return QueryClassification(
            intent=QueryIntent.DEFINITIONAL,
            confidence=0.85,
            explanation="Query asks for medical definitions or general explanations.",
        )
    if _CLINICAL.search(query):
        return QueryClassification(
            intent=QueryIntent.CLINICAL,
            confidence=0.85,
            explanation="Query pertains to clinical management, guidelines, or drug treatments.",
        )
    if _RELATIONAL.search(query):
        return QueryClassification(
            intent=QueryIntent.RELATIONAL,
            confidence=0.80,
            explanation="Query inquires about relationships between entities (e.g., lobes, seizure types).",
        )
    return QueryClassification(
        intent=QueryIntent.CLINICAL,
        confidence=0.60,
        explanation="General medical query.",
    )
