"""
Query-Adaptive Retrieval Scorer: computes S_vector, S_graph, and S_eeg for sub-queries.
"""
import re
from src.routing.schemas import RoutingScores

SCORE_SYSTEM = """You are a retrieval router for an EEG-based epilepsy reasoning system.
Given ONE sub-query, rate from 0.0 to 1.0 how much each knowledge source is needed to answer it.
Scores are independent (they do not need to sum to 1).

- s_vector: unstructured medical literature (guidelines, StatPearls, ILAE): definitions, diagnostic criteria,
  treatments, mechanisms, typical presentations, explanations.
- s_graph: structured relations between clinical entities (brain region <-> seizure type, symptom, drug, gene,
  comorbidity, EEG pattern <-> syndrome): "which / what is associated with / affected by / part of".
- s_eeg: evidence from THIS patient's own EEG recording (waveform, spikes, rhythmicity, seizure detection).
  Keep it near 0 unless the sub-query refers to this patient's own signal. A general question that merely
  mentions EEG (for example "what EEG patterns occur in absence seizures") does NOT need the patient's EEG.

Grounding rule: if the sub-query asks whether this patient's EEG supports, suggests or indicates a diagnosis
or syndrome, the signal must be interpreted against knowledge, so s_graph and s_vector must also be high.
A pure pattern-detection question about the signal ("does this EEG show rhythmic spiking") needs the signal
only, so s_graph and s_vector stay low.

Calibrated examples (the numbers show the expected scale)
- "What is epilepsy?" -> s_vector=0.92, s_graph=0.24, s_eeg=0.03
- "What brain regions are associated with focal seizures?" -> s_vector=0.62, s_graph=0.94, s_eeg=0.05
- "Does this EEG support focal epilepsy?" -> s_vector=0.72, s_graph=0.91, s_eeg=0.97
- "Does this patient's EEG show rhythmic spiking?" -> s_vector=0.15, s_graph=0.15, s_eeg=0.95
- "What are first-line drugs for focal seizures?" -> s_vector=0.90, s_graph=0.55, s_eeg=0.02"""

_DOMAIN = re.compile(r"epilep|seizure|\beeg\b|ictal|brain|lobe|spike|encephal", re.I)
_PATIENT_SIGNAL = re.compile(
    r"\bthis patient|\bthe patient|\bmy patient|\bpatient'?s\b|"
    r"\b(this|the|my|his|her|their)\s+(eeg|recording|signal|trace|scan|segment|window)\b", re.I)
_SIGNAL_TERMS = re.compile(r"spik|rhythm|waveform|discharge|burst|channel|amplitude|frequency|artifact|slowing", re.I)
_DIAGNOSTIC = re.compile(r"support|suggest|indicat|consistent with|diagnos|classif|confirm|consistent|evidence of", re.I)

_GRAPH_ENTITY = re.compile(r"region|lobe|cortex|hippocamp|amygdala|gene|drug|medication", re.I)
_GRAPH_RELATION = re.compile(r"associated|affect|related|linked|connect|comorbid|caus|symptom|part of|type of|typical of|seen in|occur in", re.I)
_GRAPH_PATTERN = re.compile(r"characteristic|finding|pattern|feature", re.I)

_VEC_DEFINE = re.compile(r"what is|what are|define|definition|explain|describe", re.I)
_VEC_CLINICAL = re.compile(r"guideline|criteria|treat|therapy|management|dose|prognos|first-line|diagnostic", re.I)
_VEC_TYPICAL = re.compile(r"typically|usually|commonly|characteristic|finding|feature", re.I)
_QUESTION = re.compile(r"\b(what|which|how|why)\b", re.I)


def _clip(v: float) -> float:
    return round(min(1.0, max(0.03, v)), 2)


def heuristic_scores(sub_query: str) -> RoutingScores:
    """Calculate heuristic suitability scores for Vector, Graph, and EEG evidence sources."""
    q = sub_query
    about_patient = bool(_PATIENT_SIGNAL.search(q))
    diagnostic = about_patient and bool(_DIAGNOSTIC.search(q)) and bool(_DOMAIN.search(q))

    # --- EEG: only when the query is about this patient's own signal ---------
    s_eeg = 0.0
    if about_patient:
        s_eeg = 0.65 + (0.30 if _SIGNAL_TERMS.search(q) else 0.0) + (0.30 if diagnostic else 0.0)

    # --- Graph: entity/relation questions ------------------------------------
    s_graph = 0.0
    s_graph += 0.50 if _GRAPH_ENTITY.search(q) else 0.0
    s_graph += 0.35 if _GRAPH_RELATION.search(q) else 0.0
    s_graph += 0.25 if _GRAPH_PATTERN.search(q) else 0.0

    # --- Vector: literature-style questions ----------------------------------
    s_vec = 0.30 if _DOMAIN.search(q) else 0.0
    s_vec += 0.50 if _VEC_DEFINE.search(q) else 0.0
    s_vec += 0.40 if _VEC_CLINICAL.search(q) else 0.0
    s_vec += 0.25 if _VEC_TYPICAL.search(q) else 0.0
    s_vec += 0.25 if (_QUESTION.search(q) and _DOMAIN.search(q)) else 0.0

    # --- Grounding rule: diagnostic question about the patient's EEG ---------
    if diagnostic:
        s_graph = max(s_graph, 0.90)
        s_vec = max(s_vec, 0.70)

    return RoutingScores(s_vector=_clip(s_vec), s_graph=_clip(s_graph), s_eeg=_clip(s_eeg))
