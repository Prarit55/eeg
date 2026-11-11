import argparse
from typing import List

from src.eeg.pipeline import DUMMY_EDF_PATH, EEGPipeline
from src.knowledge.document_loader import SAMPLE_PDF_PATH
from src.knowledge.knowledge_graph import OPTIMUS_KG_PATH, EpilepsyKnowledgeGraph
from src.knowledge.vector_store import MedicalVectorStore
from src.routing.patient_context import HistoryExtractor
from src.routing.router import QueryRouter

TEST_QUERY = ("Does this patient's EEG show rhythmic spiking, "
              "and what brain regions are typically affected by focal seizures?")
SAMPLE_HISTORY = ("The patient is a 34-year-old male with head trauma at age 20, "
                  "on carbamazepine, two prior unprovoked seizures.")


def main(edf: str, pdf: str, triplets: str, tau: float, preview: bool, history: str, extractor: str) -> None:
    # --- Instantiate Modules 1-4 (heavy work is deferred until needed) ------
    eeg = EEGPipeline()
    vector = MedicalVectorStore()
    graph = EpilepsyKnowledgeGraph(triplets)
    router = QueryRouter(tau=tau, extractor=HistoryExtractor(backend=extractor))
    context = router.extractor.extract(history) if history.strip() else None

    print(f"Knowledge graph: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges")
    if context:
        print(f"Patient context : {context.summary()}   [extracted by: {context.source}]")
        for flag in context.eeg_flags():
            print(f"  EEG note      : {flag}")

    # --- Query understanding + adaptive scoring (question only) --------------
    print(f"\nMaster query: {TEST_QUERY}")
    result = router.route(TEST_QUERY, context)
    d = result.decomposition
    print(f"\nis_complex : {d.is_complex}\nreasoning  : {d.reasoning}")
    print("sub_queries:")
    for i, q in enumerate(d.sub_queries, 1):
        print(f"  [{i}] {q}")

    print(f"\n--- Routing scores (tau = {tau}) ---")
    for i, route in enumerate(result.routes, 1):
        s = route.scores
        print(f"\n[{i}] {route.sub_query}")
        print(f"    s_vector={s.s_vector:.2f}  s_graph={s.s_graph:.2f}  s_eeg={s.s_eeg:.2f}")
        print(f"    -> would activate: {', '.join(route.activated)}")
        if route.retrieval_query != route.sub_query:
            print(f"    vector query enriched with history: {route.retrieval_query}")
    print(f"\nModules needed for this query overall: {', '.join(result.all_activated())}"
          f"  (of Vector, Graph, EEG)")

    if not preview:
        return

    # --- Optional: actually run the activated retrievers ---------------------
    print("\n--- Retrieval preview ---")
    n_chunks = vector.build([pdf]) if "Vector" in {m for r in result.routes for m in r.activated} else 0
    if n_chunks:
        print(f"Vector store built with {n_chunks} chunks.")
    for i, route in enumerate(result.routes, 1):
        print(f"\n[{i}] {route.sub_query}")
        if "Vector" in route.activated:
            for chunk in vector.retrieve_text(route.retrieval_query, k=3):
                print(f"    [vector] {chunk[:140].replace(chr(10), ' ')}...")
        if "Graph" in route.activated:
            # start entities: from the question, plus those found in the patient's history
            seed_text = route.sub_query + " " + (result.patient_context.entity_text() if result.patient_context else "")
            entities: List[str] = graph.find_entities(seed_text)[:4]
            for ent in entities:
                for line in graph.retrieve_subgraph(ent, hops=1)[:4]:
                    print(f"    [graph:{ent}] {line}")
        if "EEG" in route.activated:
            out = eeg.analyze(edf)
            print(f"    [eeg] windows={int(out['n_windows'])}  mean P(seizure)={out['seizure_prob'].mean():.3f}  "
                  f"z_patient shape={tuple(out['z_patient'].shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", default=DUMMY_EDF_PATH, help="CHB-MIT .edf file (dummy EEG used if missing)")
    ap.add_argument("--pdf", default=SAMPLE_PDF_PATH, help="Medical PDF (sample PDF generated if default is missing)")
    ap.add_argument("--triplets", default=OPTIMUS_KG_PATH, help="Triplet CSV [head, relation, tail]")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--history", default=SAMPLE_HISTORY, help="Patient medical history text ('' for none)")
    ap.add_argument("--extractor", default="auto", choices=["auto", "ner", "llm", "lexicon"],
                    help="History extractor: auto = local NER -> LLM -> keyword lists")
    ap.add_argument("--preview", action="store_true", help="Run the activated retrievers too")
    a = ap.parse_args()
    main(a.edf, a.pdf, a.triplets, a.tau, a.preview, a.history, a.extractor)
