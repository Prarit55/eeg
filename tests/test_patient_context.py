"""
Unit tests for HistoryExtractor, deidentification, and PatientContext schemas.
"""
import unittest
import warnings

from langchain_core.runnables import RunnableLambda

from src.routing.patient_context import (
    HistoryExtraction,
    HistoryExtractor,
    PatientContext,
    deidentify,
    separate_history,
)

TEXT = (
    "Mr. Rao, MRN 5521. 34-year-old male with head trauma at age 20, prior focal seizures, "
    "on carbamazepine 400 mg and clobazam."
)


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda _msgs: self.result)


class TestPatientContext(unittest.TestCase):
    def test_ner_mapping_and_grounding(self):
        preds = [
            {"entity_group": "Medication", "word": "carbamazepine 400 mg", "score": 0.99},
            {"entity_group": "Disease_disorder", "word": "head trauma", "score": 0.95},
            {"entity_group": "Disease_disorder", "word": "focal seizures", "score": 0.93},
            {"entity_group": "Disease_disorder", "word": "diabetes", "score": 0.99},
            {"entity_group": "Medication", "word": "clobazam", "score": 0.30},
        ]
        ex = HistoryExtractor(backend="ner")
        ex._ner = lambda text: preds
        c = ex.extract(TEXT)
        self.assertEqual(c.source, "ner")
        self.assertIn("carbamazepine", c.medications)
        self.assertIn("clobazam", c.medications)
        self.assertEqual(c.conditions, ["head trauma"])
        self.assertNotIn("diabetes", c.conditions)
        self.assertEqual(c.seizure_history, ["focal seizures"])
        self.assertEqual((c.age, c.sex), (34, "male"))
        self.assertNotIn("Rao", c.notes)
        self.assertNotIn("5521", c.notes)

    def test_llm_backend_and_hallucination_guard(self):
        result = HistoryExtraction(
            age=34,
            sex="male",
            conditions=["head trauma", "diabetes"],
            medications=["carbamazepine", "clobazam"],
            seizure_history=["focal seizures"],
        )
        c = HistoryExtractor(backend="llm", llm=FakeLLM(result)).extract(TEXT)
        self.assertEqual(c.source, "llm")
        self.assertEqual(c.age, 34)
        self.assertEqual(c.conditions, ["head trauma"])

    def test_fallback_chain(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ex = HistoryExtractor(backend="auto", llm=None)
            ex._ner_failed = True
            c = ex.extract(TEXT)
        self.assertEqual(c.source, "lexicon")
        self.assertIn("carbamazepine", c.medications)

    def test_separate_history(self):
        mixed = "The patient is a 25yo female with temporal lobe epilepsy. What are the best medications?"
        hist, q = separate_history(mixed)
        self.assertTrue(len(hist) > 0)
        self.assertIn("What are the best medications", q)


if __name__ == "__main__":
    unittest.main()
