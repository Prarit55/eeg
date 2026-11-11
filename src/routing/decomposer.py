"""
Constrained query decomposition into atomic, self-contained subqueries.
"""
import re
from typing import List

from src.routing.schemas import QueryDecomposition

_SPLIT_AND = re.compile(
    r"\s*[,;]?\s+and\s+(?=(?:what|which|how|why|does|do|is|are|can|where|when)\b)", re.IGNORECASE
)

DECOMPOSE_SYSTEM = """You analyse questions for an EEG-based epilepsy reasoning system.
Decide whether the user's master query is COMPLEX: it combines two or more distinct information needs
(for example one about this patient's own EEG and another about general medical knowledge).
If it is complex, split it into minimal, self-contained, atomic sub-queries that can each be answered
independently, keeping the original wording where possible. If it is not complex, return the original
query as the only sub-query.

Examples
- "What is epilepsy?" -> is_complex=false, sub_queries=["What is epilepsy?"]
- "Does this EEG support focal epilepsy, and what EEG characteristics are associated with focal epilepsy?"
  -> is_complex=true, sub_queries=["Does this EEG support focal epilepsy?",
     "What EEG characteristics are associated with focal epilepsy?"]"""


def heuristic_decompose(query: str) -> QueryDecomposition:
    """Split complex query into atomic subqueries via punctuation and conjunction analysis."""
    parts: List[str] = []
    for sentence in re.split(r"(?<=\?)\s+", query.strip()):
        parts.extend(_SPLIT_AND.split(sentence))
    parts = [p.strip(" ,;?") for p in parts if p.strip(" ,;?")]
    parts = [p[0].upper() + p[1:] + "?" for p in parts]
    complex_ = len(parts) > 1
    return QueryDecomposition(
        is_complex=complex_,
        reasoning=(
            "Query joins several distinct questions; split into atomic sub-queries."
            if complex_ else "Single information need."
        ),
        sub_queries=parts or [query],
    )
