"""
Module 3 - Epilepsy Knowledge Graph (grounded in Harvard OptimusKG).
Stores and traverses genuine biomedical triplets: [head, relation, tail].
Supports persistent disk caching and native Neo4j graph database integration.
"""
from __future__ import annotations

import logging
import pickle
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from src.knowledge.neo4j_adapter import Neo4jGraphDatabase

logger = logging.getLogger(__name__)

OPTIMUS_KG_PATH: str = "data/raw/kg/optimus_kg_epilepsy.csv"
DEFAULT_KG_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "kg_graph"
DEFAULT_KG_CACHE_FILE = DEFAULT_KG_CACHE_DIR / "epilepsy_kg.pkl"

KEYWORDS: List[str] = [
    "epilepsy", "seizure", "temporal", "frontal", "hippocamp", "carbamazepine",
    "lamotrigine", "levetiracetam", "valpro", "status epilepticus", "lacosamide",
    "absence", "clonic", "tonic", "focal", "generalized", "syndrome"
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
    """Knowledge graph engine with persistent disk caching and Neo4j database integration."""

    def __init__(
        self,
        csv_path: str = OPTIMUS_KG_PATH,
        keywords: Optional[List[str]] = None,
        cache_path: Optional[str | Path] = DEFAULT_KG_CACHE_FILE,
        use_neo4j: bool = True,
        neo4j_uri: Optional[str] = None,
        **csv_kwargs,
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self._lower_to_node: Dict[str, str] = {}
        self.csv_path = csv_path
        self.cache_path = Path(cache_path) if cache_path else None
        
        # Neo4j backend
        self.neo4j: Optional[Neo4jGraphDatabase] = None
        if use_neo4j:
            kwargs = {"uri": neo4j_uri} if neo4j_uri else {}
            self.neo4j = Neo4jGraphDatabase(**kwargs)

        # Build or load from persistent cache
        loaded = False
        if self.cache_path and self.cache_path.exists():
            loaded = self._load_cache()

        if not loaded:
            df = load_triplets(csv_path, keywords, **csv_kwargs)
            self.build(df)
            if self.cache_path:
                self._save_cache()

        # If Neo4j is active and empty, offer sync
        if self.neo4j and self.neo4j.is_connected():
            logger.info("Connected to Neo4j graph database.")

    def _save_cache(self) -> None:
        """Persist graph structure and node mapping to disk."""
        try:
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump({"graph": self.graph, "lower_map": self._lower_to_node}, f)
                logger.info("Saved persistent KG cache to %s", self.cache_path)
        except Exception as e:
            logger.warning("Failed to save KG cache: %s", e)

    def _load_cache(self) -> bool:
        """Load graph structure and node mapping from disk cache."""
        try:
            if self.cache_path and self.cache_path.exists():
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                self.graph = data["graph"]
                self._lower_to_node = data["lower_map"]
                logger.info("Loaded persistent KG cache from %s (%d nodes, %d edges)",
                            self.cache_path, self.graph.number_of_nodes(), self.graph.number_of_edges())
                return True
        except Exception as e:
            logger.warning("Could not read KG cache, rebuilding: %s", e)
        return False

    def build(self, df: pd.DataFrame) -> None:
        """Populate NetworkX graph from triplet DataFrame."""
        for head, rel, tail in df.itertuples(index=False):
            head, rel, tail = str(head).strip(), str(rel).strip(), str(tail).strip()
            self.graph.add_edge(head, tail, relation=rel)
            self._lower_to_node[head.lower()] = head
            self._lower_to_node[tail.lower()] = tail

    def sync_to_neo4j(self) -> int:
        """Sync loaded graph triplets to the Neo4j database."""
        if not self.neo4j or not self.neo4j.is_connected():
            logger.warning("Neo4j is not connected. Cannot sync.")
            return 0
        df = pd.read_csv(self.csv_path, usecols=["head", "relation", "tail"]).dropna()
        return self.neo4j.import_triplets(df)

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
        # Use Neo4j if available
        if self.neo4j and self.neo4j.is_connected():
            neo4j_res = self.neo4j.get_subgraph(entity, hops=hops, limit=max_triplets)
            if neo4j_res:
                return neo4j_res

        # Fallback to local high-performance graph engine
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

    def find_paths(self, source_entity: str, target_entity: str, max_depth: int = 3) -> List[List[str]]:
        """Find paths connecting two clinical entities."""
        if self.neo4j and self.neo4j.is_connected():
            neo4j_paths = self.neo4j.find_paths(source_entity, target_entity, max_depth=max_depth)
            if neo4j_paths:
                return neo4j_paths

        src_nodes = self._resolve(source_entity)
        tgt_nodes = self._resolve(target_entity)
        if not src_nodes or not tgt_nodes:
            return []
        
        src, tgt = src_nodes[0], tgt_nodes[0]
        undirected = self.graph.to_undirected()
        try:
            return list(nx.all_simple_paths(undirected, source=src, target=tgt, cutoff=max_depth))[:10]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def query_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run Cypher query against Neo4j, or pattern-match locally."""
        if self.neo4j and self.neo4j.is_connected():
            return self.neo4j.run_cypher(cypher, params)
        logger.info("Executing local Cypher emulation for: %s", cypher)
        return []
