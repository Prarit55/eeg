"""
Unit tests for QueryRouter, intent classification, and decomposition.
"""
import unittest

from src.routing.classifier import QueryIntent, classify_query
from src.routing.decomposer import heuristic_decompose
from src.routing.patient_context import HistoryExtractor
from src.routing.router import QueryRouter
from src.routing.scorer import heuristic_scores


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter(tau=0.5, extractor=HistoryExtractor(backend="lexicon"))

    def test_decomposition(self):
        complex_q = (
            "Does this patient's EEG show rhythmic spiking, "
            "and what brain regions are typically affected by focal seizures?"
        )
        decomp = heuristic_decompose(complex_q)
        self.assertTrue(decomp.is_complex)
        self.assertEqual(len(decomp.sub_queries), 2)

        simple_q = "What is epilepsy?"
        decomp_simple = heuristic_decompose(simple_q)
        self.assertFalse(decomp_simple.is_complex)
        self.assertEqual(len(decomp_simple.sub_queries), 1)

    def test_classifier(self):
        c1 = classify_query("What is epilepsy?")
        self.assertEqual(c1.intent, QueryIntent.DEFINITIONAL)

        c2 = classify_query("Does this patient's EEG show rhythmic spiking?")
        self.assertEqual(c2.intent, QueryIntent.EEG_SPECIFIC)

    def test_routing_scores(self):
        scores = heuristic_scores("What is epilepsy?")
        self.assertGreaterEqual(scores.s_vector, 0.5)
        self.assertLess(scores.s_eeg, 0.5)

        eeg_scores = heuristic_scores("Does this patient's EEG show rhythmic spiking?")
        self.assertGreaterEqual(eeg_scores.s_eeg, 0.5)


if __name__ == "__main__":
    unittest.main()
