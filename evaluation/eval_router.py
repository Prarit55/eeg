"""
Routing evaluation and history invariance benchmarking.
Directly measures routing accuracy and stability.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing.patient_context import HistoryExtractor
from src.routing.router import QueryRouter

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmarks" / "test_cases.json"

HISTORY = "The patient is a 34-year-old male with head trauma at age 20, on carbamazepine, two prior unprovoked seizures."


def run_evaluation(tau: float = 0.5) -> int:
    router = QueryRouter(tau=tau, extractor=HistoryExtractor(backend="lexicon"))
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Filter single-intent cases for direct score testing
    single_cases = [c for c in cases if c["category"] != "complex"]
    wrong = 0
    print(f"\n--- Routing Benchmark (tau = {tau}) ---")
    print(f"{'Query':<60} {'V':>5} {'G':>5} {'E':>5}  Activated -> Expected")
    for c in single_cases:
        q = c["query"]
        expected = set(c["expected_sources"])
        scores = router.score([q])[0]
        got = set(router.activated_modules(scores))
        ok = got == expected
        wrong += not ok
        print(f"{q[:59]:<60} {scores.s_vector:5.2f} {scores.s_graph:5.2f} {scores.s_eeg:5.2f}  "
              f"{sorted(got)} {'OK' if ok else 'MISMATCH expected ' + str(sorted(expected))}")

    print(f"\nRouting Accuracy: {len(single_cases) - wrong}/{len(single_cases)} matches expected.")

    # History invariance check
    bad = 0
    print("\n--- Patient History Invariance Benchmark ---")
    for c in single_cases:
        q = c["query"]
        expected = set(c["expected_sources"])
        routed = router.route(f"{HISTORY} {q}")
        got = set(routed.routes[0].activated)
        has_ctx = routed.patient_context is not None
        ok = (got == expected) and has_ctx
        bad += not ok
        if not ok:
            print(f"FAILED history invariance for '{q}': got {sorted(got)}")

    print(f"History Invariance: {len(single_cases) - bad}/{len(single_cases)} unchanged.")
    return wrong + bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()
    exit(1 if run_evaluation(args.tau) else 0)
