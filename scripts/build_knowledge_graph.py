"""
Build and inspect knowledge graph statistics.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.knowledge_graph import OPTIMUS_KG_PATH, EpilepsyKnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Inspector")
    parser.add_argument("--triplets", default=OPTIMUS_KG_PATH, help="Path to triplets CSV")
    args = parser.parse_args()

    kg = EpilepsyKnowledgeGraph(args.triplets)
    print("=" * 60)
    print("EPILEPSY KNOWLEDGE GRAPH METRICS")
    print("=" * 60)
    print(f"Total Nodes: {kg.graph.number_of_nodes()}")
    print(f"Total Edges: {kg.graph.number_of_edges()}")
    print("\nSample Nodes:")
    for n in list(kg.graph.nodes())[:10]:
        print(f"  - {n}")
    print("\nSample Relations:")
    for u, v, d in list(kg.graph.edges(data=True))[:8]:
        print(f"  - ({u}) --[{d.get('relation')}]--> ({v})")
    print("=" * 60)


if __name__ == "__main__":
    main()
