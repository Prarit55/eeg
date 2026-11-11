"""
Multimodal Evidence Fusion Layer.
"""
from src.fusion.aggregator import (
    DocumentEvidence,
    EEGEvidence,
    EvidenceAggregator,
    EvidenceBundle,
    TripletEvidence,
)
from src.fusion.context_formatter import format_multimodal_context

__all__ = [
    "EvidenceBundle",
    "EEGEvidence",
    "TripletEvidence",
    "DocumentEvidence",
    "EvidenceAggregator",
    "format_multimodal_context",
]
