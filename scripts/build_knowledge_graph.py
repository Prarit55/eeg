"""
Build and inspect knowledge graph statistics, persistent caching, and Neo4j connectivity.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.knowledge_graph import OPTIMUS_KG_PATH, EpilepsyKnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Inspector & Persistent Cache Builder")
    parser.add_argument("--triplets", default=OPTIMUS_KG_PATH, help="Path to triplets CSV")
    parser.add_argument("--sync-neo4j", action="store_true", help="Sync loaded triplets to Neo4j database")
    args = parser.parse_args()

    print("=" * 80)
    print("EPILEPSY KNOWLEDGE GRAPH DATABASE INSPECTOR")
    print("=" * 80)
    kg = EpilepsyKnowledgeGraph(args.triplets)

    print(f"Total Nodes: {kg.graph.number_of_nodes()}")
    print(f"Total Edges: {kg.graph.number_of_edges()}")

    neo4j_status = "Connected" if (kg.neo4j and kg.neo4j.is_connected()) else "Offline (Using local graph engine)"
    print(f"Neo4j Status: {neo4j_status}")

    if args.sync_neo4j:
        if kg.neo4j and kg.neo4j.is_connected():
            print("Syncing triplets to Neo4j...")
            count = kg.sync_to_neo4j()
            print(f"Successfully synced {count} triplets to Neo4j.")
        else:
            print("Cannot sync: Neo4j is offline.")

    print("\nSample Nodes:")
    for n in list(kg.graph.nodes())[:10]:
        print(f"  - {n}")

    print("\nSample Subgraph Retrieval for 'carbamazepine':")
    subgraph = kg.retrieve_subgraph("carbamazepine", hops=1, max_triplets=5)
    for s in subgraph:
        print(f"  [KG] {s}")

    print("\nSample Path Finding between 'temporal lobe' and 'carbamazepine':")
    paths = kg.find_paths("temporal lobe", "carbamazepine", max_depth=3)
    if paths:
        for p in paths[:3]:
            print(f"  Path: {' -> '.join(p)}")
    else:
        print("  No path within depth 3.")

    print("=" * 80)
    print("Knowledge Graph build and cache verification completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
