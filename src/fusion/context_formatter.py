"""
Multimodal context formatter for medical LLM reasoning.
"""
from typing import List
from src.fusion.aggregator import EvidenceBundle


def format_multimodal_context(bundle: EvidenceBundle) -> str:
    """Format an EvidenceBundle into a clean, structured context string with citation markers."""
    sections: List[str] = []

    # 1. Patient Context
    if bundle.patient_context and not bundle.patient_context.is_empty():
        ctx_lines = [f"[Patient-Context] {bundle.patient_context.summary()}"]
        for flag in bundle.patient_context.eeg_flags():
            ctx_lines.append(f"  - Clinical Flag: {flag}")
        sections.append("\n".join(ctx_lines))

    # 2. EEG Signal Evidence
    if bundle.eeg:
        sections.append(
            f"[EEG-Evidence]\n"
            f"  - Windows Analyzed: {bundle.eeg.n_windows}\n"
            f"  - Mean P(seizure): {bundle.eeg.mean_seizure_prob:.2f}\n"
            f"  - Max P(seizure): {bundle.eeg.max_seizure_prob:.2f}\n"
            f"  - Spiking Detected: {'Yes' if bundle.eeg.rhythmic_spiking_detected else 'No'}\n"
            f"  - Signal Interpretation: {bundle.eeg.summary}"
        )

    # 3. Knowledge Graph Triples
    if bundle.graph and bundle.graph.triplets:
        kg_lines = ["[Knowledge-Graph-Triplets]"]
        for i, t in enumerate(bundle.graph.triplets, 1):
            kg_lines.append(f"  [KG-{i}] {t}")
        sections.append("\n".join(kg_lines))

    # 4. Vector Documents / Literature
    if bundle.vector and bundle.vector.chunks:
        doc_lines = ["[Medical-Literature-Excerpts]"]
        for i, c in enumerate(bundle.vector.chunks, 1):
            clean_chunk = c.strip().replace("\n", " ")
            doc_lines.append(f"  [Doc-{i}] {clean_chunk}")
        sections.append("\n".join(doc_lines))

    if not sections:
        return "No external evidence retrieved (answering from internal clinical knowledge only)."

    return "\n\n".join(sections)
