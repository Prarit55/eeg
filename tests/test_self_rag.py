"""
Unit tests for Self-RAG reflection critic, generator, and loop.
"""
import unittest

from src.fusion.aggregator import EvidenceAggregator
from src.self_rag.critique import SelfRAGCritic
from src.self_rag.generator import EvidenceGroundedGenerator


class TestSelfRAG(unittest.TestCase):
    def setUp(self):
        self.critic = SelfRAGCritic(confidence_threshold=0.6)
        self.generator = EvidenceGroundedGenerator(self.critic)
        self.aggregator = EvidenceAggregator()

    def test_critique_with_evidence(self):
        bundle = self.aggregator.build_bundle(
            query="What is focal epilepsy?",
            graph_triplets=["Temporal Lobe is associated with Focal Seizure"],
            vector_chunks=["Focal seizures originate within networks limited to one hemisphere."],
        )
        res = self.generator.generate(bundle)
        self.assertGreaterEqual(res.confidence, 0.6)
        self.assertTrue(res.critique.is_relevant)
        self.assertFalse(res.critique.needs_refinement)
        self.assertTrue(len(res.citations) > 0)

    def test_critique_without_evidence(self):
        bundle = self.aggregator.build_bundle(query="Unknown question")
        res = self.generator.generate(bundle)
        self.assertTrue(res.critique.needs_refinement)
        self.assertLess(res.confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
