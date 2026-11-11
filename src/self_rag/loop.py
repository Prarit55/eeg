"""
Full Query-Adaptive Self-RAG Execution Loop.
Connects routing, selective retrieval, multimodal fusion, and self-reflection critique.
"""
from typing import Any, Dict, List, Optional

from src.eeg.pipeline import EEGPipeline
from src.fusion.aggregator import EvidenceAggregator, EvidenceBundle
from src.knowledge.knowledge_graph import EpilepsyKnowledgeGraph
from src.knowledge.vector_store import MedicalVectorStore
from src.routing.patient_context import PatientContext
from src.routing.router import QueryRouter
from src.routing.schemas import RoutedQuery
from src.self_rag.critique import SelfRAGCritic
from src.self_rag.generator import EvidenceGroundedGenerator, EvidenceGroundedResponse


class SelfRAGLoop:
    """Orchestrates query decomposition, adaptive retrieval, evidence fusion, critique, and generation."""

    def __init__(
        self,
        router: QueryRouter,
        eeg_pipeline: EEGPipeline,
        vector_store: MedicalVectorStore,
        knowledge_graph: EpilepsyKnowledgeGraph,
        max_refinement_steps: int = 1,
    ) -> None:
        self.router = router
        self.eeg = eeg_pipeline
        self.vector = vector_store
        self.graph = knowledge_graph
        self.max_refinement_steps = max_refinement_steps
        self.aggregator = EvidenceAggregator()
        self.generator = EvidenceGroundedGenerator(SelfRAGCritic())

    def run(
        self,
        master_query: str,
        edf_path: Optional[str] = None,
        patient_history: str = "",
    ) -> Dict[str, Any]:
        """Execute the end-to-end adaptive Self-RAG reasoning pipeline."""
        context = (
            self.router.extractor.extract(patient_history)
            if patient_history.strip() else None
        )

        # 1. Routing & decomposition
        routed: RoutedQuery = self.router.route(master_query, context)
        sub_results: List[EvidenceGroundedResponse] = []

        for route in routed.routes:
            sub_q = route.sub_query
            activated = route.activated

            # 2. Selective Retrieval based on activated modules
            eeg_out = None
            if "EEG" in activated:
                eeg_out = self.eeg.analyze(edf_path)

            kg_triplets: List[str] = []
            seed_entities: List[str] = []
            if "Graph" in activated:
                seed_text = sub_q + " " + (context.entity_text() if context else "")
                seed_entities = self.graph.find_entities(seed_text)[:4]
                for ent in seed_entities:
                    kg_triplets.extend(self.graph.retrieve_subgraph(ent, hops=1)[:4])

            doc_chunks: List[str] = []
            if "Vector" in activated:
                try:
                    doc_chunks = self.vector.retrieve_text(route.retrieval_query, k=2)
                except Exception:
                    pass

            # 3. Evidence Fusion
            bundle: EvidenceBundle = self.aggregator.build_bundle(
                query=sub_q,
                patient_context=context,
                eeg_analysis=eeg_out,
                graph_triplets=kg_triplets,
                seed_entities=seed_entities,
                vector_chunks=doc_chunks,
            )

            # 4. Self-RAG Reflection & Generation
            response = self.generator.generate(bundle)
            sub_results.append(response)

        return {
            "master_query": master_query,
            "routed": routed,
            "sub_results": sub_results,
            "activated_modules": routed.all_activated(),
        }
