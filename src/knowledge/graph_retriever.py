"""
Graph retrieval and multi-hop traversal utilities.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import networkx as nx

from src.knowledge.knowledge_graph import EpilepsyKnowledgeGraph


class GraphRetriever:
    """Helper class to query an EpilepsyKnowledgeGraph with entity extraction and path search."""

    def __init__(self, graph: EpilepsyKnowledgeGraph) -> None:
        self.kg = graph

    def get_relevant_triplets(self, query_text: str, hops: int = 1, max_triplets: int = 10) -> List[str]:
        """Extract entities from query_text and retrieve their k-hop subgraphs."""
        entities = self.kg.find_entities(query_text)
        triplets: List[str] = []
        for ent in entities[:4]:
            lines = self.kg.retrieve_subgraph(ent, hops=hops, max_triplets=max_triplets)
            for line in lines:
                if line not in triplets:
                    triplets.append(line)
        return triplets[:max_triplets]

    def find_paths(self, source_entity: str, target_entity: str, max_depth: int = 3) -> List[List[str]]:
        """Find simple paths between two entities in the knowledge graph."""
        src_nodes = self.kg._resolve(source_entity)
        tgt_nodes = self.kg._resolve(target_entity)
        if not src_nodes or not tgt_nodes:
            return []
        
        src, tgt = src_nodes[0], tgt_nodes[0]
        # Undirected view for traversal
        undirected = self.kg.graph.to_undirected()
        try:
            return list(nx.all_simple_paths(undirected, source=src, target=tgt, cutoff=max_depth))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
