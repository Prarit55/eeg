"""
Patient context (medical history) - kept SEPARATE from the question.

Why separate?
* The router must score the QUESTION only. History text mentions "the patient", drugs and conditions, which
  would otherwise trigger the EEG/graph sources for questions that are not about them.
* History is evidence about the patient, like the EEG, so it is used later at three points:
    1. retrieval enrichment : `enrich(sub_query)` adds history terms to the vector query
    2. graph seeding        : `entity_text()` gives extra start entities for `retrieve_subgraph`
    3. fusion / Self-RAG    : `summary()` becomes its own labelled evidence block (next stage)
* Privacy: history is protected health information. `deidentify()` strips obvious identifiers before any
  text can reach an external LLM. It is best-effort regex, NOT a guarantee; use a vetted de-identification
  tool (e.g. Presidio) or a local model for real patient data.

Parsing here is lexicon-based so the project runs offline. Replace `from_text` with an LLM or clinical NER
extractor later; the `PatientContext` schema is the stable interface.
"""
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# De-identification (best effort)
# --------------------------------------------------------------------------- #
_PHI_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b(?:MRN|ID|patient id|record no\.?)\s*[:#]?\s*[\w-]+", re.I), "[ID]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)\d{3}[\s-]?\d{3,4}(?!\w)"), "[PHONE]"),
    (re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"), "[NAME]"),
    (re.compile(r"\b(?:name|patient name)\s*[:=]\s*[A-Z][\w .'-]+", re.I), "[NAME]"),
]


def deidentify(text: str) -> str:
    """Replace emails, IDs, dates, phone numbers and titled names with placeholders (best effort)."""
    for pat, repl in _PHI_PATTERNS:
        text = pat.sub(repl, text)
    return text


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# anti-seizure drugs / meds
_MEDICATIONS = [
    "carbamazepine", "oxcarbazepine", "lamotrigine", "levetiracetam", "valproic acid", "valproate",
    "phenytoin", "phenobarbital", "topiramate", "lacosamide", "zonisamide", "clobazam", "clonazepam",
    "lorazepam", "diazepam", "midazolam", "ethosuximide",
]
_CONDITIONS = [
    "head trauma", "head injury", "traumatic brain injury", "stroke", "meningitis", "encephalitis",
    "brain tumor", "brain tumour", "hippocampal sclerosis", "cortical dysplasia", "migraine",
    "neonatal hypoxia", "family history of epilepsy", "intellectual disability",
]
_SEIZURE_TERMS = [
    "focal seizures", "generalized tonic-clonic seizures", "tonic-clonic seizures", "absence seizures",
    "febrile seizures", "febrile seizure", "status epilepticus", "unprovoked seizures", "nocturnal seizures",
    "myoclonic seizures",
]
# Drug classes known to add beta activity to the background EEG (widely taught; confirm with a clinician).
_BETA_DRUGS = {"clobazam", "clonazepam", "lorazepam", "diazepam", "midazolam", "phenobarbital"}


class PatientContext(BaseModel):
    age: Optional[int] = Field(default=None, ge=0, le=120)
    sex: Optional[str] = None
    conditions: List[str] = Field(default_factory=list, description="Past/comorbid conditions and risk factors.")
    medications: List[str] = Field(default_factory=list, description="Current (anti-seizure) medications.")
    seizure_history: List[str] = Field(default_factory=list, description="Known seizure types / events.")
    notes: str = Field(default="", description="De-identified source text the fields were extracted from.")
    source: str = Field(default="lexicon", description="Which extractor produced the fields: ner, llm or lexicon.")

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_text(cls, text: str, extractor: Optional["HistoryExtractor"] = None) -> "PatientContext":
        """Parse free-text history. Uses `extractor` (NER / LLM) or, by default, the offline keyword lists."""
        return (extractor or HistoryExtractor(backend="lexicon")).extract(text)

    # -- views -------------------------------------------------------------- #
    def is_empty(self) -> bool:
        return not (self.age or self.sex or self.conditions or self.medications or self.seizure_history)

    def summary(self) -> str:
        """Compact text for the evidence-fusion block."""
        parts: List[str] = []
        demo = " ".join(x for x in [f"{self.age}-year-old" if self.age else "", self.sex or ""] if x)
        if demo:
            parts.append(demo)
        if self.conditions:
            parts.append("history: " + ", ".join(self.conditions))
        if self.seizure_history:
            parts.append("seizures: " + ", ".join(self.seizure_history))
        if self.medications:
            parts.append("medications: " + ", ".join(self.medications))
        return "; ".join(parts) if parts else "no structured history"

    def entity_text(self) -> str:
        """Terms to look up in the knowledge graph (conditions, seizure types, drugs)."""
        return " ".join(self.conditions + self.seizure_history + self.medications)

    def enrich(self, sub_query: str) -> str:
        """Sub-query + history terms, for the vector search only (never sent to the router/LLM)."""
        terms = self.conditions + self.seizure_history
        if self.age:
            terms.insert(0, "adult" if self.age >= 18 else "paediatric")
        return f"{sub_query} Patient context: {', '.join(terms)}." if terms else sub_query

    def eeg_flags(self) -> List[str]:
        """History items that can change how the EEG should be read (shown to the reasoning step)."""
        return [f"{d}: may increase background beta activity" for d in self.medications if d in _BETA_DRUGS]


# --------------------------------------------------------------------------- #
# Extractors: local biomedical NER, LLM with structured output, keyword lists
# --------------------------------------------------------------------------- #
NER_MODEL = "d4data/biomedical-ner-all"   # local HuggingFace token-classification model (no PHI leaves the machine)
_SEIZURE_WORD = re.compile(r"seizure|convuls|ictal|status epilepticus|absence|tonic|clonic|myoclon", re.I)
_DOSE = re.compile(r"\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b.*$", re.I)


class HistoryExtraction(BaseModel):
    """Schema the LLM must fill (same fields as PatientContext, minus notes/source)."""
    age: Optional[int] = Field(default=None, ge=0, le=120, description="Age in years, null if not stated.")
    sex: Optional[Literal["male", "female"]] = Field(default=None, description="null if not stated.")
    conditions: List[str] = Field(default_factory=list, description="Illnesses, injuries, risk factors.")
    medications: List[str] = Field(default_factory=list, description="Generic drug names, no doses.")
    seizure_history: List[str] = Field(default_factory=list, description="Seizure types or events.")


# No curly braces except the {text} variable (they would be read as template variables).
EXTRACT_SYSTEM = """You extract structured facts from a patient's medical history for an epilepsy decision-support system.
Rules:
- Extract ONLY what is explicitly written. Never infer, guess or add diagnoses.
- conditions: past or current illnesses, injuries and risk factors. Not seizure types, not drugs.
- seizure_history: seizure types or events (for example focal seizures, status epilepticus).
- medications: drug names only, generic and lowercase, without doses.
- age: integer years. sex: male or female. Use null when not stated.
- The text may contain placeholders such as [NAME] or [DATE]. Ignore them."""


def _lexicon_extract(clean: str) -> Dict[str, Any]:
    """Deterministic offline extraction from the keyword lists (also used to boost recall of the others)."""
    low = clean.lower()
    age = None
    m = re.search(r"(\d{1,3})\s*[- ]?(?:year|yr)s?[- ]?old|\bage[d]?\s*(\d{1,3})\b|\b(\d{1,3})\s?(?:y/?o|yo)\b", low)
    if m:
        age = int(next(g for g in m.groups() if g))
    sex = None
    m = re.search(r"\b(male|female|man|woman|boy|girl)\b", low)
    if m:
        sex = "male" if m.group(1) in ("male", "man", "boy") else "female"

    def found(terms: List[str]) -> List[str]:
        out: List[str] = []
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", low) and not any(t in o for o in out):
                out.append(t)
        return out

    return {"age": age, "sex": sex, "conditions": found(_CONDITIONS),
            "medications": found(_MEDICATIONS), "seizure_history": found(_SEIZURE_TERMS)}


def _clean_term(term: str) -> str:
    return _DOSE.sub("", term.lower().replace("##", "")).strip(" ,.;:-")


def _grounded(term: str, low_text: str) -> bool:
    """Guard against hallucinated entities: the term, or all of its words, must appear in the source text."""
    if term in low_text:
        return True
    words = [w for w in re.findall(r"[a-z]+", term) if len(w) >= 3]
    return bool(words) and all(w in low_text for w in words)


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    for t in items:
        if t and not any(t == o or t in o for o in out):
            out = [o for o in out if o not in t] + [t]
    return out


class HistoryExtractor:
    """Turn free-text history into a PatientContext.

    backend
      "auto"    : local NER -> LLM -> keyword lists (first one that works)
      "ner"     : local biomedical NER model only (privacy-friendly: runs on your machine)
      "llm"     : LLM with structured output only (sends the de-identified text to the model provider)
      "lexicon" : keyword lists only (offline, fixed vocabulary)

    Safeguards for every backend
      * text is de-identified first (see `deidentify`)
      * entities not found in the source text are dropped (no hallucinated history)
      * results are unioned with the keyword lists (`merge_lexicon`) so known epilepsy drugs/conditions are
        not missed; a backend that fails falls back to the next one with a warning.
    """

    def __init__(
        self,
        backend: Literal["auto", "ner", "llm", "lexicon"] = "auto",
        llm: Optional[Any] = None,
        ner_model: str = NER_MODEL,
        min_score: float = 0.5,
        merge_lexicon: bool = True,
    ) -> None:
        self.backend, self.llm, self.ner_model = backend, llm, ner_model
        self.min_score, self.merge_lexicon = min_score, merge_lexicon
        self._ner: Optional[Any] = None
        self._ner_failed = False
        self._llm_chain: Optional[Any] = None

    # -- backends ----------------------------------------------------------- #
    def _ner_pipeline(self):
        if self._ner is None:
            if self._ner_failed:
                raise RuntimeError("NER model could not be loaded earlier")
            try:
                from transformers import pipeline
                self._ner = pipeline("token-classification", model=self.ner_model, aggregation_strategy="simple")
            except Exception:
                self._ner_failed = True
                raise
        return self._ner

    def _run_ner(self, clean: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"conditions": [], "medications": [], "seizure_history": []}
        for ent in self._ner_pipeline()(clean):
            if float(ent.get("score", 1.0)) < self.min_score:
                continue
            label, term = ent.get("entity_group", ""), _clean_term(ent.get("word", ""))
            if not term:
                continue
            if label == "Medication":
                out["medications"].append(term)
            elif label in ("Disease_disorder", "Sign_symptom", "Clinical_event") and _SEIZURE_WORD.search(term):
                out["seizure_history"].append(term)
            elif label == "Disease_disorder":
                out["conditions"].append(term)
        return out

    def _run_llm(self, clean: str) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("no LLM provided")
        if self._llm_chain is None:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([("system", EXTRACT_SYSTEM), ("human", "{text}")])
            self._llm_chain = prompt | self.llm.with_structured_output(HistoryExtraction, method="function_calling")
        return self._llm_chain.invoke({"text": clean}).model_dump()

    # -- public ------------------------------------------------------------- #
    def extract(self, text: str) -> PatientContext:
        clean = deidentify(text)
        low = clean.lower()
        base = _lexicon_extract(clean)

        order = {"auto": ["ner", "llm"], "ner": ["ner"], "llm": ["llm"], "lexicon": []}[self.backend]
        got: Optional[Dict[str, Any]] = None
        used = "lexicon"
        for name in order:
            try:
                got = self._run_ner(clean) if name == "ner" else self._run_llm(clean)
                used = name
                break
            except Exception as e:  # missing model, no network, no key, bad output ...
                warnings.warn(f"HistoryExtractor: {name} backend unavailable ({type(e).__name__}: {e}); falling back.")

        fields: Dict[str, List[str]] = {}
        for key in ("conditions", "medications", "seizure_history"):
            extra = [_clean_term(t) for t in (got or {}).get(key, [])]
            extra = [t for t in extra if t and _grounded(t, low)]                 # drop hallucinations
            fields[key] = _dedupe((base[key] if (self.merge_lexicon or not got) else []) + extra)

        age = (got or {}).get("age") or base["age"]
        sex = (got or {}).get("sex") or base["sex"]
        return PatientContext(age=age, sex=sex, notes=clean.strip(), source=used, **fields)


# --------------------------------------------------------------------------- #
# Split "history + question" pasted into one string
# --------------------------------------------------------------------------- #
_QUESTION_START = re.compile(
    r"^(does|do|did|is|are|was|were|can|could|should|what|which|how|why|where|when|who|show|find|explain|"
    r"tell|describe|list|summari[sz]e)\b", re.I)


def separate_history(text: str) -> Tuple[str, str]:
    """Split pasted text into (history, question).

    A sentence is part of the question if it ends with '?' or starts with a question/command word.
    Everything else is history. If no sentence looks like a question, the whole text is the question.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]
    q = [s for s in sentences if s.endswith("?") or _QUESTION_START.match(s)]
    if not q or len(q) == len(sentences):
        return "", text.strip()
    h = [s for s in sentences if s not in q]
    return " ".join(h), " ".join(q)
