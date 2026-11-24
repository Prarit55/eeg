"""
Unit tests for FastAPI backend service endpoints.
"""
import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app


class TestAPIRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")

    def test_query_endpoint(self):
        payload = {
            "query": "What is epilepsy?",
            "patient_history": "30-year-old female on lamotrigine.",
        }
        resp = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("master_query", data)
        self.assertIn("activated_modules", data)
        self.assertIn("results", data)
        self.assertTrue(len(data["results"]) > 0)

    def test_eeg_analyze_endpoint(self):
        payload = {
            "edf_path": "data/raw/eeg/chb01/chb01_03.edf",
        }
        resp = self.client.post("/api/v1/eeg/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("n_windows", data)
        self.assertIn("mean_seizure_prob", data)
        self.assertIn("max_seizure_prob", data)
        self.assertIn("rhythmic_spiking", data)


if __name__ == "__main__":
    unittest.main()
