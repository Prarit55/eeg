"""
Retrieval Efficiency Benchmark:
Compares Query-Adaptive Retrieval against a Non-Adaptive baseline (which activates all sources for every query).
Objective 7 of Minor Project Synopsis:
"Evaluate retrieval efficiency, evidence quality, and answer correctness against non-adaptive baselines."
"""
import json
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing.patient_context import HistoryExtractor
from src.routing.router import QueryRouter

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmarks" / "test_cases.json"


def evaluate_retrieval_efficiency(tau: float = 0.5) -> None:
    router = QueryRouter(tau=tau, extractor=HistoryExtractor(backend="lexicon"))
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total_baseline_activations = len(cases) * 3  # Non-adaptive always activates [Vector, Graph, EEG]
    adaptive_activations = 0

    print("=" * 80)
    print("RETRIEVAL EFFICIENCY EVALUATION: Query-Adaptive vs. Non-Adaptive Baseline")
    print("=" * 80)
    print(f"{'ID':<4} {'Query':<45} {'Adaptive Sources':<20} {'Modules Saved':<12}")
    print("-" * 80)

    for c in cases:
        routed = router.route(c["query"])
        activated = routed.all_activated()
        count = len(activated)
        adaptive_activations += count
        saved = 3 - count
        print(f"#{c['id']:<3} {c['query'][:44]:<45} {str(activated):<20} {saved:<12}")

    reduction = ((total_baseline_activations - adaptive_activations) / total_baseline_activations) * 100
    print("=" * 80)
    print(f"Total non-adaptive activations (all sources): {total_baseline_activations}")
    print(f"Total query-adaptive activations:            {adaptive_activations}")
    print(f"Retrieval efficiency gain (reduction):      {reduction:.1f}% fewer redundant calls")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_retrieval_efficiency()
