"""
Document loading and PDF generation helpers for medical literature and guidelines.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE_PDF_PATH: str = "data/sample_epilepsy.pdf"

_SAMPLE_TEXT: List[str] = [
    "Epilepsy is a chronic neurological disorder characterised by an enduring predisposition to generate "
    "unprovoked seizures. A seizure is a transient occurrence of signs or symptoms due to abnormal excessive "
    "or synchronous neuronal activity in the brain.",
    "Focal seizures originate within networks limited to one hemisphere. Temporal lobe epilepsy is the most "
    "common focal epilepsy in adults and frequently involves the hippocampus, amygdala and parahippocampal "
    "gyrus. Frontal lobe seizures are often brief, occur from sleep, and show prominent motor features such "
    "as hypermotor or tonic posturing.",
    "On scalp EEG, interictal epileptiform discharges include spikes (20-70 ms) and sharp waves (70-200 ms), "
    "often followed by a slow wave. Generalised 3 Hz spike-and-wave discharges are the electrographic "
    "hallmark of childhood absence epilepsy. Rhythmic evolving activity in the theta or alpha range may mark "
    "the onset of a focal electrographic seizure.",
    "Diagnosis relies on clinical history, EEG and neuroimaging. MRI with epilepsy protocol is recommended "
    "to identify structural causes such as hippocampal sclerosis, focal cortical dysplasia, tumours and "
    "vascular malformations. Prolonged video-EEG monitoring helps localise the seizure onset zone.",
    "First-line treatment of focal seizures includes carbamazepine, lamotrigine and levetiracetam. "
    "Generalised tonic-clonic seizures are commonly treated with valproate, lamotrigine or levetiracetam. "
    "Drug-resistant epilepsy, defined as failure of two appropriately chosen and tolerated antiseizure "
    "medications, should prompt referral for surgical evaluation.",
    "Status epilepticus is a neurological emergency defined by a seizure lasting more than five minutes or "
    "recurrent seizures without recovery. Initial management uses intravenous or intramuscular "
    "benzodiazepines, followed by a second-line antiseizure medication such as levetiracetam or fosphenytoin.",
]


def create_sample_pdf(path: str = SAMPLE_PDF_PATH) -> str:
    """Generate a small multi-page sample PDF (needs `reportlab`). Returns the path."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    _, height = A4
    y = height - 60
    c.setFont("Helvetica", 11)
    for para in _SAMPLE_TEXT * 2:  # repeat so the PDF spans more than one page
        for line in textwrap.wrap(para, 95):
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 60
            c.drawString(50, y, line)
            y -= 16
        y -= 10
    c.save()
    return path


def load_and_split_documents(
    pdf_paths: List[str],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Load and recursively split PDF documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())
    return splitter.split_documents(docs)
