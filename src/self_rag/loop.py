"""
Full Query-Adaptive Self-RAG Execution Loop.
Connects routing, selective retrieval, multimodal fusion, self-reflection critique,
and iterative retrieval revision / refinement loop.
"""
from typing import Any, Dict, List, Optional, Set

from src.eeg.pipeline import EEGPipeline
from src.fusion.aggregator import EvidenceAggregator, EvidenceBundle
from src.knowledge.knowledge_graph import EpilepsyKnowledgeGraph
from src.knowledge.vector_store import MedicalVectorStore
from src.routing.patient_context import PatientContext
from src.routing.router import QueryRouter
from src.routing.schemas import RoutedQuery, SubQueryRoute
from src.self_rag.critique import SelfRAGCritic
from src.self_rag.generator import EvidenceGroundedGenerator, EvidenceGroundedResponse


class SelfRAGLoop:
    """Orchestrates query decomposition, adaptive retrieval, evidence fusion, critique, and refinement."""

    def __init__(
        self,
        router: QueryRouter,
        eeg_pipeline: EEGPipeline,
        vector_store: MedicalVectorStore,
        knowledge_graph: EpilepsyKnowledgeGraph,
        max_refinement_steps: int = 2,
    ) -> None:
        self.router = router
        self.eeg = eeg_pipeline
        self.vector = vector_store
        self.graph = knowledge_graph
        self.max_refinement_steps = max_refinement_steps
        self.aggregator = EvidenceAggregator()
        self.generator = EvidenceGroundedGenerator(SelfRAGCritic())

    def _retrieve_evidence(
        self,
        sub_q: str,
        retrieval_query: str,
        activated: List[str],
        context: Optional[PatientContext],
        edf_path: Optional[str],
        graph_hops: int = 1,
    ) -> EvidenceBundle:
        """Execute selective retrieval based on activated channels."""
        # 1. EEG
        eeg_out = None
        if "EEG" in activated:
            eeg_out = self.eeg.analyze(edf_path)

        # 2. Knowledge Graph
        kg_triplets: List[str] = []
        seed_entities: List[str] = []
        if "Graph" in activated:
            seed_text = sub_q + " " + (context.entity_text() if context else "")
            seed_entities = self.graph.find_entities(seed_text)[:4]
            for ent in seed_entities:
                kg_triplets.extend(self.graph.retrieve_subgraph(ent, hops=graph_hops)[:4])

        # 3. Vector Guidelines
        doc_chunks: List[str] = []
        if "Vector" in activated:
            try:
                doc_chunks = self.vector.retrieve_text(retrieval_query, k=3)
            except Exception:
                pass

        return self.aggregator.build_bundle(
            query=sub_q,
            patient_context=context,
            eeg_analysis=eeg_out,
            graph_triplets=kg_triplets,
            seed_entities=seed_entities,
            vector_chunks=doc_chunks,
        )

    def run(
        self,
        master_query: str,
        edf_path: Optional[str] = None,
        patient_history: str = "",
    ) -> Dict[str, Any]:
        """Execute the end-to-end adaptive Self-RAG reasoning pipeline with iterative refinement."""
        context = (
            self.router.extractor.extract(patient_history)
            if patient_history.strip() else None
        )

        # 1. Routing & decomposition
        routed: RoutedQuery = self.router.route(master_query, context)
        sub_results: List[EvidenceGroundedResponse] = []
        refinement_logs: List[str] = []

        for route in routed.routes:
            sub_q = route.sub_query
            current_activated = list(route.activated)
            graph_hops = 1

            # Initial retrieval & generation
            bundle = self._retrieve_evidence(
                sub_q=sub_q,
                retrieval_query=route.retrieval_query,
                activated=current_activated,
                context=context,
                edf_path=edf_path,
                graph_hops=graph_hops,
            )
            response = self.generator.generate(bundle)

            # --- Self-RAG Reflection & Iterative Refinement Loop ---
            step = 0
            while response.critique.needs_refinement and step < self.max_refinement_steps:
                step += 1
                revised = False
                reasons = []

                # Condition A: Vector was not active or yielded no chunks
                if "Vector" not in current_activated:
                    current_activated.append("Vector")
                    revised = True
                    reasons.append("Activated Vector guideline retrieval")
                elif not bundle.vector or not bundle.vector.chunks:
                    # Broaden retrieval query
                    route.retrieval_query = f"{sub_q} epilepsy diagnosis clinical guidelines"
                    revised = True
                    reasons.append("Broadened vector retrieval search terms")

                # Condition B: Graph was active but found 0 triplets -> expand hops or search
                if "Graph" in current_activated and (not bundle.graph or not bundle.graph.triplets):
                    graph_hops = min(3, graph_hops + 1)
                    revised = True
                    reasons.append(f"Expanded graph traversal to {graph_hops} hops")
                elif "Graph" not in current_activated and route.scores.s_graph >= (self.router.tau - 0.20):
                    current_activated.append("Graph")
                    revised = True
                    reasons.append("Activated borderline Graph source")

                if not revised:
                    break

                log_entry = f"Step {step} refinement for '{sub_q[:40]}...': {', '.join(reasons)}"
                refinement_logs.append(log_entry)

                # Re-retrieve with updated decisions
                bundle = self._retrieve_evidence(
                    sub_q=sub_q,
                    retrieval_query=route.retrieval_query,
                    activated=current_activated,
                    context=context,
                    edf_path=edf_path,
                    graph_hops=graph_hops,
                )
                new_response = self.generator.generate(bundle)
                new_response.critique.critique_notes = (
                    f"[Refined Step {step}]: {', '.join(reasons)}. {new_response.critique.critique_notes}"
                )
                response = new_response

            sub_results.append(response)

        return {
            "master_query": master_query,
            "routed": routed,
            "sub_results": sub_results,
            "activated_modules": routed.all_activated(),
            "refinement_logs": refinement_logs,
        }
