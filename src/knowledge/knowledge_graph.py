"""
Module 3 - Epilepsy Knowledge Graph (grounded in Harvard OptimusKG).
Stores and traverses genuine biomedical triplets: [head, relation, tail].
Loads extracted OptimusKG epilepsy connections (drugs, phenotypes, diseases, brain anatomy).
"""
from __future__ import annotations

import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

OPTIMUS_KG_PATH: str = "data/raw/kg/optimus_kg_epilepsy.csv"
KEYWORDS: List[str] = [
    "epilepsy", "seizure", "temporal", "frontal", "hippocamp", "carbamazepine",
    "lamotrigine", "levetiracetam", "valpro", "status epilepticus"
]


def load_triplets(
    csv_path: str = OPTIMUS_KG_PATH,
    keywords: Optional[List[str]] = None,
    head_col: str = "head",
    rel_col: str = "relation",
    tail_col: str = "tail",
) -> pd.DataFrame:
    """Load OptimusKG triplet CSV and optionally filter by clinical keywords."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"OptimusKG file not found at {csv_path}. "
            "Run `python scripts/build_optimus_kg.py` to extract genuine OptimusKG connections."
        )

    df = pd.read_csv(path, usecols=[head_col, rel_col, tail_col])
    df = df.rename(columns={head_col: "head", rel_col: "relation", tail_col: "tail"}).dropna()

    if keywords:
        pattern = "|".join(re.escape(k) for k in keywords)
        mask = (
            df["head"].astype(str).str.contains(pattern, case=False, regex=True)
            | df["tail"].astype(str).str.contains(pattern, case=False, regex=True)
        )
        return df[mask].reset_index(drop=True)

    return df.reset_index(drop=True)


class EpilepsyKnowledgeGraph:
    """In-memory MultiDiGraph loaded from Harvard OptimusKG with natural-language subgraph retrieval."""

    def __init__(
        self,
        csv_path: str = OPTIMUS_KG_PATH,
        keywords: Optional[List[str]] = None,
        **csv_kwargs,
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self._lower_to_node: Dict[str, str] = {}
        self.csv_path = csv_path
        self.build(load_triplets(csv_path, keywords, **csv_kwargs))

    def build(self, df: pd.DataFrame) -> None:
        for head, rel, tail in df.itertuples(index=False):
            head, rel, tail = str(head).strip(), str(rel).strip(), str(tail).strip()
            self.graph.add_edge(head, tail, relation=rel)
            self._lower_to_node[head.lower()] = head
            self._lower_to_node[tail.lower()] = tail

    # -- entity resolution -------------------------------------------------- #
    def _resolve(self, entity: str) -> List[str]:
        """Exact (case-insensitive) match first, otherwise substring matches."""
        key = entity.strip().lower()
        if key in self._lower_to_node:
            return [self._lower_to_node[key]]
        return [n for low, n in self._lower_to_node.items() if key in low]

    def find_entities(self, text: str) -> List[str]:
        """Graph nodes mentioned in free text (handles simple plurals, longest names first)."""
        low = text.lower()
        hits = [n for key, n in self._lower_to_node.items() if key in low or (key + "s") in low]
        return sorted(hits, key=len, reverse=True)

    # -- retrieval ---------------------------------------------------------- #
    def retrieve_subgraph(self, entity: str, hops: int = 1, max_triplets: int = 50) -> List[str]:
        """All triplets within `hops` of `entity`, formatted as natural language statements."""
        start = set(self._resolve(entity))
        visited, frontier = set(start), set(start)
        edges = {}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for u, v, k, d in chain(
                    self.graph.out_edges(n, keys=True, data=True),
                    self.graph.in_edges(n, keys=True, data=True),
                ):
                    edges[(u, v, k)] = d["relation"]
                    nxt.update((u, v))
            frontier = nxt - visited
            visited |= nxt
        sentences = sorted({f"{u} {rel.replace('_', ' ')} {v}" for (u, v, _), rel in edges.items()})
        return sentences[:max_triplets]
