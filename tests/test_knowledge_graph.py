"""
Unit tests for EpilepsyKnowledgeGraph and GraphRetriever.
"""
import unittest

from src.knowledge.graph_retriever import GraphRetriever
from src.knowledge.knowledge_graph import EpilepsyKnowledgeGraph, load_triplets


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.kg = EpilepsyKnowledgeGraph()
        self.retriever = GraphRetriever(self.kg)

    def test_graph_has_nodes_and_edges(self):
        # OptimusKG epilepsy subgraph contains >2,000 entities and >7,000 edges
        self.assertGreater(self.kg.graph.number_of_nodes(), 2000)
        self.assertGreater(self.kg.graph.number_of_edges(), 7000)

    def test_find_entities(self):
        text = "The patient has temporal lobe epilepsy and seizures on carbamazepine."
        entities = self.kg.find_entities(text)
        self.assertTrue(any("temporal lobe" in e.lower() for e in entities))
        self.assertTrue(any("seizure" in e.lower() for e in entities))
        self.assertTrue(any("carbamazepine" in e.lower() for e in entities))

    def test_subgraph_retrieval(self):
        sub = self.kg.retrieve_subgraph("Temporal Lobe", hops=1)
        self.assertGreater(len(sub), 0)
        self.assertTrue(any("temporal lobe" in s.lower() for s in sub))

        drug_sub = self.kg.retrieve_subgraph("Carbamazepine", hops=1)
        self.assertGreater(len(drug_sub), 0)
        self.assertTrue(any("carbamazepine" in s.lower() for s in drug_sub))


if __name__ == "__main__":
    unittest.main()
