"""
OptimusKG Epilepsy Knowledge Graph Extractor.
Extracts genuine biomedical relations from Harvard Zitnik Lab's OptimusKG:
- Epilepsy diseases & syndromes (partial epilepsy, temporal lobe epilepsy, generalized epilepsy, status epilepticus, etc.)
- Antiseizure medications and neuroactive drugs (indications, contraindications, off-label uses)
- Seizure manifestations and clinical phenotypes (auras, automatisms, motor/non-motor signs)
- Brain neuroanatomy and structural relations (temporal lobe, frontal lobe, hippocampus, amygdala, etc.)

Saves extracted project-relevant triplets to `data/raw/kg/optimus_kg_epilepsy.csv`.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import polars as pl
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Dataverse file IDs for OptimusKG modular parquet files
DATAVERSE_FILES = {
    "disease_nodes": "https://dataverse.harvard.edu/api/access/datafile/14220892",      # 2.83 MB
    "drug_nodes": "https://dataverse.harvard.edu/api/access/datafile/14220910",         # 31.43 MB
    "phenotype_nodes": "https://dataverse.harvard.edu/api/access/datafile/14220905",    # 0.93 MB
    "anatomy_nodes": "https://dataverse.harvard.edu/api/access/datafile/14220913",      # 1.04 MB
    "drug_disease_edges": "https://dataverse.harvard.edu/api/access/datafile/14220904", # 1.27 MB
    "disease_disease_edges": "https://dataverse.harvard.edu/api/access/datafile/14220916", # 0.17 MB
    "disease_phenotype_edges": "https://dataverse.harvard.edu/api/access/datafile/14220915", # 1.29 MB
    "anatomy_anatomy_edges": "https://dataverse.harvard.edu/api/access/datafile/14220889", # 0.07 MB
}

DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "kg" / "optimus_kg_epilepsy.csv"


def download_parquet(url: str, label: str) -> pl.DataFrame:
    """Download a parquet file from Harvard Dataverse into Polars DataFrame."""
    print(f"Downloading OptimusKG {label}...")
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return pl.read_parquet(io.BytesIO(resp.content))


def extract_names_from_properties(df: pl.DataFrame) -> Dict[str, str]:
    """Extract id -> name mapping from node DataFrame properties column."""
    id_map: Dict[str, str] = {}
    for row in df.iter_rows(named=True):
        props = row.get("properties")
        if isinstance(props, dict) and "name" in props and props["name"]:
            name = str(props["name"]).strip()
            # Clean up trailing/leading brackets or quotes
            name = name.replace("'", "").replace('"', "")
            id_map[row["id"]] = name
    return id_map


def build_optimus_epilepsy_kg(output_path: Path | str = DEFAULT_OUTPUT_CSV) -> pd.DataFrame:
    """Extract project-specific epilepsy connections from OptimusKG."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTRACTING PROJECT-RELEVANT SUBGRAPH FROM HARVARD OPTIMUSKG")
    print("=" * 80)

    # 1. Disease nodes (Filter for epilepsy, seizure, convulsion, encephalopathy)
    df_dis = download_parquet(DATAVERSE_FILES["disease_nodes"], "Disease Nodes")
    dis_id_map = extract_names_from_properties(df_dis)
    epilepsy_dis_ids: Set[str] = {
        nid for nid, name in dis_id_map.items()
        if any(term in name.lower() for term in [
            "epilep", "seizur", "convuls", "status epilepticus", "dravet", "lennox-gastaut", "infantile spasm"
        ])
    }
    print(f"Identified {len(epilepsy_dis_ids)} epilepsy & seizure disease nodes in OptimusKG.")

    # 2. Drug nodes
    df_drug = download_parquet(DATAVERSE_FILES["drug_nodes"], "Drug Nodes")
    drug_id_map = extract_names_from_properties(df_drug)

    # 3. Phenotype nodes
    df_pheno = download_parquet(DATAVERSE_FILES["phenotype_nodes"], "Phenotype Nodes")
    pheno_id_map = extract_names_from_properties(df_pheno)

    # 4. Anatomy nodes (Filter for brain structures)
    df_anat = download_parquet(DATAVERSE_FILES["anatomy_nodes"], "Anatomy Nodes")
    anat_id_map = extract_names_from_properties(df_anat)
    brain_terms = ["brain", "lobe", "cortex", "hippocamp", "amygdala", "thalam", "cerebr", "temporal", "frontal", "occipital", "parietal", "neuron"]
    brain_anat_ids: Set[str] = {
        nid for nid, name in anat_id_map.items()
        if any(bt in name.lower() for bt in brain_terms)
    }
    print(f"Identified {len(brain_anat_ids)} brain neuroanatomy nodes in OptimusKG.")

    triplets: List[Tuple[str, str, str]] = []

    # --- Edge Set 1: Drug <-> Epilepsy Disease (Indications & Contraindications) ---
    df_dd = download_parquet(DATAVERSE_FILES["drug_disease_edges"], "Drug-Disease Edges")
    epilepsy_dd = df_dd.filter(pl.col("to").is_in(list(epilepsy_dis_ids)))
    for row in epilepsy_dd.iter_rows(named=True):
        drug = drug_id_map.get(row["from"])
        dis = dis_id_map.get(row["to"])
        rel = row["relation"].lower().replace("_", " ")
        if drug and dis:
            # Standardize relation
            if "indication" in rel:
                rel_str = "is indicated for"
            elif "contraindication" in rel:
                rel_str = "is contraindicated for"
            elif "off label" in rel:
                rel_str = "off-label for"
            else:
                rel_str = rel
            triplets.append((drug.title(), rel_str, dis.title()))
    print(f"Added {len(triplets)} Drug-Disease connections.")

    # --- Edge Set 2: Disease <-> Disease Hierarchy (Subtypes and Parents) ---
    count_before = len(triplets)
    df_dis_dis = download_parquet(DATAVERSE_FILES["disease_disease_edges"], "Disease-Disease Edges")
    # Both from and to in epilepsy or from in epilepsy
    epilepsy_dis_dis = df_dis_dis.filter(pl.col("from").is_in(list(epilepsy_dis_ids)) | pl.col("to").is_in(list(epilepsy_dis_ids)))
    for row in epilepsy_dis_dis.iter_rows(named=True):
        d1 = dis_id_map.get(row["from"])
        d2 = dis_id_map.get(row["to"])
        if d1 and d2:
            triplets.append((d1.title(), "is a subtype of", d2.title()))
    print(f"Added {len(triplets) - count_before} Disease-Disease hierarchy connections.")

    # --- Edge Set 3: Disease <-> Phenotypes (Symptoms, Auras, Signs) ---
    count_before = len(triplets)
    df_dp = download_parquet(DATAVERSE_FILES["disease_phenotype_edges"], "Disease-Phenotype Edges")
    epilepsy_dp = df_dp.filter(pl.col("from").is_in(list(epilepsy_dis_ids)))
    for row in epilepsy_dp.iter_rows(named=True):
        dis = dis_id_map.get(row["from"])
        pheno = pheno_id_map.get(row["to"])
        if dis and pheno:
            triplets.append((dis.title(), "manifests as", pheno.title()))
    print(f"Added {len(triplets) - count_before} Disease-Phenotype symptom connections.")

    # --- Edge Set 4: Brain Anatomy <-> Brain Anatomy Hierarchy ---
    count_before = len(triplets)
    df_aa = download_parquet(DATAVERSE_FILES["anatomy_anatomy_edges"], "Anatomy-Anatomy Edges")
    brain_aa = df_aa.filter(pl.col("from").is_in(list(brain_anat_ids)) & pl.col("to").is_in(list(brain_anat_ids)))
    for row in brain_aa.iter_rows(named=True):
        a1 = anat_id_map.get(row["from"])
        a2 = anat_id_map.get(row["to"])
        if a1 and a2:
            triplets.append((a1.title(), "is part of", a2.title()))
    print(f"Added {len(triplets) - count_before} Brain Anatomy structural connections.")

    # Remove duplicates
    triplets_df = pd.DataFrame(triplets, columns=["head", "relation", "tail"]).drop_duplicates().reset_index(drop=True)
    triplets_df.to_csv(out_p, index=False)

    print("=" * 80)
    print(f"OPTIMUS-KG EXTRACTION COMPLETE!")
    print(f"Total Unique Triplets: {len(triplets_df)}")
    print(f"Total Unique Entities: {len(set(triplets_df['head']).union(set(triplets_df['tail'])))}")
    print(f"Saved to: {out_p}")
    print("=" * 80)
    return triplets_df


if __name__ == "__main__":
    build_optimus_epilepsy_kg()
