"""
CLI script to run the end-to-end Query-Adaptive Multimodal Self-RAG pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eeg.pipeline import DUMMY_EDF_PATH, EEGPipeline
from src.knowledge.document_loader import SAMPLE_PDF_PATH
from src.knowledge.knowledge_graph import OPTIMUS_KG_PATH, EpilepsyKnowledgeGraph
from src.knowledge.vector_store import MedicalVectorStore
from src.routing.patient_context import HistoryExtractor
from src.routing.router import QueryRouter
from src.self_rag.loop import SelfRAGLoop

TEST_QUERY = (
    "Does this patient's EEG show rhythmic spiking, "
    "and what brain regions are typically affected by focal seizures?"
)
SAMPLE_HISTORY = (
    "The patient is a 34-year-old male with head trauma at age 20, "
    "on carbamazepine, two prior unprovoked seizures."
)


def main():
    parser = argparse.ArgumentParser(description="Query-Adaptive Multimodal Self-RAG CLI Runner")
    parser.add_argument("--query", default=TEST_QUERY, help="Master query string")
    parser.add_argument("--history", default=SAMPLE_HISTORY, help="Patient clinical history")
    parser.add_argument("--edf", default=DUMMY_EDF_PATH, help="EDF file path")
    parser.add_argument("--pdf", default=SAMPLE_PDF_PATH, help="Medical guideline PDF path")
    parser.add_argument("--triplets", default=OPTIMUS_KG_PATH, help="Knowledge graph triplet CSV")
    parser.add_argument("--tau", type=float, default=0.5, help="Routing threshold tau")
    args = parser.parse_args()

    print("=" * 80)
    print("QUERY-ADAPTIVE MULTIMODAL SELF-RAG PIPELINE")
    print("=" * 80)
    print(f"Master Query: {args.query}\n")

    eeg = EEGPipeline()
    vector = MedicalVectorStore()
    graph = EpilepsyKnowledgeGraph(args.triplets)
    router = QueryRouter(tau=args.tau, extractor=HistoryExtractor(backend="lexicon"))

    loop = SelfRAGLoop(
        router=router,
        eeg_pipeline=eeg,
        vector_store=vector,
        knowledge_graph=graph,
    )

    result = loop.run(args.query, edf_path=args.edf, patient_history=args.history)
    routed = result["routed"]

    print(f"Query Decomposition (is_complex={routed.decomposition.is_complex}):")
    for i, sq in enumerate(routed.decomposition.sub_queries, 1):
        print(f"  [{i}] {sq}")

    print("\nActivated Sources per Sub-Query:")
    for i, r in enumerate(routed.routes, 1):
        s = r.scores
        print(f"  [{i}] {r.sub_query}")
        print(f"      Scores: s_vector={s.s_vector:.2f}, s_graph={s.s_graph:.2f}, s_eeg={s.s_eeg:.2f}")
        print(f"      Activated: {r.activated}")

    print("\n" + "=" * 80)
    print("REASONING & EVIDENCE-GROUNDED OUTPUTS")
    print("=" * 80)
    for i, res in enumerate(result["sub_results"], 1):
        print(f"\n--- Sub-Query [{i}]: {res.query} ---")
        print(f"Answer:\n{res.answer}\n")
        print(f"Citations: {res.citations}")
        print(f"Confidence: {res.confidence:.2f}")
        print(f"Critique: {res.critique.critique_notes}")


if __name__ == "__main__":
    main()
