"""
Harvard PrimeKG (Precision Medicine Knowledge Graph) Fetcher & Subgraph Extractor.
Harvard Dataverse URL: https://dataverse.harvard.edu/api/access/datafile/6180135
Ingests PrimeKG and filters disease/phenotype/anatomy/drug relations relevant to neurology & epilepsy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.knowledge.knowledge_graph import load_triplets

PRIMEKG_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/6180135"
OUTPUT_KG_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "kg" / "primekg_neurology_subgraph.csv"

NEURO_KEYWORDS = [
    "epilepsy",
    "seizure",
    "temporal lobe",
    "frontal lobe",
    "hippocampus",
    "amygdala",
    "cortex",
    "convulsion",
    "anticonvulsant",
    "status epilepticus",
    "encephalopathy",
]


def extract_neurology_subgraph(
    primekg_csv_path: str | Path,
    output_path: str | Path = OUTPUT_KG_PATH,
    keywords: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter raw PrimeKG (x_name, display_relation, y_name) to neurology-relevant relations."""
    keywords = keywords or NEURO_KEYWORDS
    print(f"Filtering PrimeKG at {primekg_csv_path} with keywords: {keywords}...")
    
    sub_df = load_triplets(
        csv_path=str(primekg_csv_path),
        keywords=keywords,
        head_col="x_name",
        rel_col="display_relation",
        tail_col="y_name",
    )
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_p, index=False)
    print(f"Extracted {len(sub_df)} neurology triplets saved to: {out_p}")
    return sub_df


def main():
    parser = argparse.ArgumentParser(description="PrimeKG Neurology Subgraph Extractor")
    parser.add_argument("--primekg-path", help="Local path to downloaded kg.csv from Harvard Dataverse")
    parser.add_argument("--output", default=str(OUTPUT_KG_PATH), help="Output CSV path for extracted subgraph")
    args = parser.parse_args()

    if args.primekg_path and Path(args.primekg_path).exists():
        extract_neurology_subgraph(args.primekg_path, args.output)
    else:
        print("=" * 80)
        print("HARVARD PRIMEKG INGESTION GUIDE")
        print("=" * 80)
        print(f"Download PrimeKG (kg.csv, ~1.5 GB) directly from Harvard Dataverse:")
        print(f"  URL: {PRIMEKG_DATAVERSE_URL}")
        print("\nOnce downloaded, run this script to extract the neurology & epilepsy subgraph:")
        print(f"  python scripts/fetch_primekg.py --primekg-path path/to/kg.csv")
        print("=" * 80)


if __name__ == "__main__":
    main()
