"""
Unit tests for EvidenceAggregator and format_multimodal_context.
"""
import unittest

from src.fusion.aggregator import EvidenceAggregator
from src.fusion.context_formatter import format_multimodal_context
from src.routing.patient_context import HistoryExtractor, PatientContext


class TestEvidenceFusion(unittest.TestCase):
    def setUp(self):
        self.aggregator = EvidenceAggregator()
        self.extractor = HistoryExtractor(backend="lexicon")

    def test_aggregation_and_formatting(self):
        ctx = self.extractor.extract("34-year-old male with focal seizures on carbamazepine.")
        bundle = self.aggregator.build_bundle(
            query="Does this patient have focal epilepsy?",
            patient_context=ctx,
            graph_triplets=["Temporal Lobe is associated with Focal Seizure"],
            vector_chunks=["Carbamazepine is indicated for focal seizures."],
        )

        self.assertTrue(bundle.has_any_evidence())
        formatted = format_multimodal_context(bundle)
        self.assertIn("[Patient-Context]", formatted)
        self.assertIn("[Knowledge-Graph-Triplets]", formatted)
        self.assertIn("[Medical-Literature-Excerpts]", formatted)


if __name__ == "__main__":
    unittest.main()
