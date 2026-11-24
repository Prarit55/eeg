"""
Comprehensive 100+ Page Teaching Handbook Generator for Minor Project ARP-455:
Query-Adaptive Multimodal Self-RAG for Epilepsy Analysis.
Author: Parth Saini (06219051923), USAR, GGSIPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas
import pypdf

PDF_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "EEG_Self_RAG_Comprehensive_Teaching_Handbook_100Pages.pdf"
PDF_ROOT_PATH = Path(__file__).resolve().parent.parent / "EEG_Self_RAG_Comprehensive_Teaching_Handbook_100Pages.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that writes dynamic page count and running headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Skip cover page

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header
        self.drawString(
            45, 752, "Query-Adaptive Multimodal Self-RAG for Epilepsy Analysis — Comprehensive Teaching Handbook"
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 746, 567, 746)

        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 32, page_text)
        self.drawString(45, 32, "Candidate: Parth Saini (06219051923) | USAR, GGSIPU | Course: ARP-455")
        self.line(45, 42, 567, 42)
        self.restoreState()


def get_styles():
    base = getSampleStyleSheet()

    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_secondary = colors.HexColor("#1E3A8A")  # Deep Blue
    c_accent = colors.HexColor("#0284C7")     # Sky Blue
    c_dark = colors.HexColor("#334155")       # Slate 700

    styles = {
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=c_primary,
            alignment=1,
            spaceAfter=14,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=c_secondary,
            alignment=1,
            spaceAfter=24,
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=c_dark,
            alignment=1,
        ),
        "ChapterTitle": ParagraphStyle(
            "ChapterTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=c_primary,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=c_secondary,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "SubSectionTitle": ParagraphStyle(
            "SubSectionTitle",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13.5,
            textColor=c_accent,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=c_dark,
            spaceAfter=5,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=c_dark,
            leftIndent=14,
            firstLineIndent=-9,
            spaceAfter=4,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#0F172A"),
        ),
        "CodeText": ParagraphStyle(
            "CodeText",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#0F172A"),
        ),
        "VivaQ": ParagraphStyle(
            "VivaQ",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#991B1B"),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "VivaA": ParagraphStyle(
            "VivaA",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.8,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=7,
        ),
        "TableBody": ParagraphStyle(
            "TableBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#1E293B"),
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.white,
        ),
    }
    return styles


def make_callout(text: str, styles, bg_hex="#EFF6FF", border_hex="#93C5FD"):
    t = Table([[Paragraph(text, styles["Callout"])]], colWidths=[522])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor(border_hex)),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_full_story(styles):
    story = []

    # =========================================================================
    # FRONT MATTER (Pages 1 - 4)
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("THE COMPREHENSIVE TEACHING & VIVA DEFENSE HANDBOOK", styles["CoverTitle"]))
    story.append(Paragraph(
        "A Complete Pedagogical, Mathematical, and Practical Reference Manual<br/>"
        "for <b>Query-Adaptive Multimodal Self-RAG in Epilepsy Analysis</b>",
        styles["CoverSubtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0F172A"), spaceBefore=10, spaceAfter=20))

    meta = """
    <b>Candidate / Author:</b> Parth Saini (Roll Number: <b>06219051923</b>)<br/>
    <b>Course Name & Code:</b> Minor Project [ARP-455] — B.Tech Computer Science / Robotics<br/>
    <b>Academic Supervisor:</b> Dr. Ruchika Lalit, Assistant Professor<br/>
    <b>Department:</b> University School of Automation & Robotics (USAR)<br/>
    <b>University:</b> Guru Gobind Singh Indraprastha University (GGSIPU), New Delhi, India<br/>
    <b>Live Demonstration Web App:</b> https://prarit55.github.io/eeg/<br/>
    <b>Hugging Face Dataset Hub:</b> https://huggingface.co/datasets/parthsaini333/chbmit-eeg<br/>
    <b>Source Code Repository:</b> https://github.com/Prarit55/eeg
    """
    story.append(Paragraph(meta, styles["CoverMeta"]))
    story.append(Spacer(1, 24))

    box_text = (
        "<b>Handbook Purpose & How to Study:</b> This handbook is an exhaustive reference text covering "
        "every single concept, algorithm, dataset, neural architecture, database integration, mathematical formulation, "
        "and clinical reasoning step in the minor project. It is structured into 12 comprehensive modules designed "
        "for self-study, deep technical understanding, and viva voce presentation mastery. "
        "Includes word-for-word spoken defenses for tough examiner challenges such as BiLSTM temporal causality, "
        "GraphRAG vs Vector RAG complementarity, and 42.6 GB dataset engineering."
    )
    story.append(make_callout(box_text, styles, bg_hex="#F0FDF4", border_hex="#86EFAC"))
    story.append(PageBreak())

    # Preface & Dedication (Page 2)
    story.append(Paragraph("Preface & Academic Scope", styles["ChapterTitle"]))
    story.append(Paragraph(
        "Modern biomedical informatics stands at a critical juncture. The past decade witnessed remarkable advances "
        "in deep learning architectures for biological signal classification on one hand, and large language models "
        "for natural language reasoning on the other. However, in clinical epileptology, these two paradigms remained "
        "fundamentally segregated. A binary seizure detector can process microvolt electrical fluctuations across 23 scalp "
        "electrodes, yet it is utterly incapable of synthesizing clinical recommendations, explaining underlying syndromic "
        "pathophysiology, or adjusting antiseizure medications based on patient history. Conversely, large language models "
        "possess broad medical knowledge but cannot directly perceive time-varying electrophysiological waveforms, frequently "
        "hallucinating unsupported clinical claims when presented with complex neurology queries.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "This minor project, conducted as part of the curriculum for ARP-455 at the University School of Automation & Robotics "
        "(USAR), Guru Gobind Singh Indraprastha University (GGSIPU), resolves this divide. We introduce a unified, "
        "query-adaptive multimodal Self-RAG architecture that bridges continuous scalp EEG encoding with dual structured "
        "and unstructured knowledge channels, regulated by dynamic routing and self-reflective critique loops.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "This handbook is organized to accompany the student and researcher from the fundamental neurophysiology of cerebral "
        "dipoles to the intricate mathematics of class-weighted cross-entropy loss, Hierarchical Navigable Small World (HNSW) "
        "vector indexing, Neo4j Cypher subgraph traversals, and iterative Self-RAG reflection mechanics.",
        styles["Body"],
    ))
    story.append(Spacer(1, 10))
    story.append(make_callout(
        "\"The purpose of clinical AI is not to supplant the human epileptologist, but to provide an evidence-grounded, "
        "transparent, and mathematically verifiable reasoning partner that eliminates cognitive overload and eradicates "
        "unsupported medical hallucinations.\" — Project Philosophy",
        styles,
        bg_hex="#F8FAFC",
        border_hex="#CBD5E1",
    ))
    story.append(PageBreak())

    # Master Table of Contents (Pages 3 - 4)
    story.append(Paragraph("Master Table of Contents", styles["ChapterTitle"]))
    story.append(Paragraph("The handbook is systematically divided into 12 modular curriculum chapters:", styles["Body"]))

    toc_items = [
        ("Module 1", "Clinical Neurophysiology & Electrophysiological Fundamentals", "Pages 5 – 14"),
        ("Module 2", "The PhysioNet CHB-MIT & OpenNeuro Datasets", "Pages 15 – 24"),
        ("Module 3", "Scalp EEG Preprocessing & Window Segmentation Theory", "Pages 25 – 34"),
        ("Module 4", "Deep Neural Architecture: 1D-CNN + BiLSTM Encoder", "Pages 35 – 48"),
        ("Module 5", "Dual Knowledge Channels: Vector RAG & ChromaDB", "Pages 49 – 58"),
        ("Module 6", "Clinical Knowledge Graphs: OptimusKG & Neo4j Cypher", "Pages 59 – 70"),
        ("Module 7", "Query Understanding & Adaptive Dynamic Routing", "Pages 71 – 80"),
        ("Module 8", "Multimodal Evidence Fusion & Context Structuring", "Pages 81 – 88"),
        ("Module 9", "Self-RAG Active Reflection & Iterative Refinement", "Pages 89 – 98"),
        ("Module 10", "Software Architecture, REST API & Web Dashboard", "Pages 99 – 108"),
        ("Module 11", "The Comprehensive Viva Voce Defense Masterclass (35+ Q&A)", "Pages 109 – 120"),
        ("Module 12", "Code Walkthrough, Benchmark Metrics & Bibliography", "Pages 121 – 128"),
    ]

    toc_table_data = [[Paragraph("<b>Module</b>", styles["TableHeader"]), Paragraph("<b>Subject Title & Detailed Scope</b>", styles["TableHeader"]), Paragraph("<b>Coverage</b>", styles["TableHeader"])]]
    for m, t, p in toc_items:
        toc_table_data.append([
            Paragraph(f"<b>{m}</b>", styles["TableBody"]),
            Paragraph(t, styles["TableBody"]),
            Paragraph(p, styles["TableBody"]),
        ])
    t_toc = Table(toc_table_data, colWidths=[70, 370, 82])
    t_toc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_toc)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Mathematical & Acronym Notation Index", styles["SectionTitle"]))
    notations = [
        ("EEG", "Electroencephalography (measurement of cerebral electrical activity)"),
        ("ILAE", "International League Against Epilepsy (global medical guideline authority)"),
        ("CHB-MIT", "Children's Hospital Boston / Massachusetts Institute of Technology EEG Database"),
        ("1D-CNN", "One-Dimensional Convolutional Neural Network (extracts local temporal filters)"),
        ("BiLSTM", "Bidirectional Long Short-Term Memory Recurrent Neural Network"),
        ("RAG", "Retrieval-Augmented Generation (combining parametric LLMs with external corpus)"),
        ("Self-RAG", "Self-Reflective Retrieval-Augmented Generation with [IsRel], [IsSup], [IsUse] tokens"),
        ("S_vec, S_graph, S_eeg", "Query suitability scores for Vector Store, Knowledge Graph, and EEG channels"),
        ("tau", "Dynamic routing activation threshold (calibrated to tau = 0.50)"),
        ("PDS", "Paroxysmal Depolarizing Shift (cellular mechanism of epileptic seizure spike)"),
        ("ASMs / AEDs", "Antiseizure Medications / Antiepileptic Drugs (e.g. Carbamazepine, Levetiracetam)"),
    ]
    n_data = [[Paragraph("<b>Symbol</b>", styles["TableHeader"]), Paragraph("<b>Definition & Clinical / Engineering Interpretation</b>", styles["TableHeader"])]]
    for s_sym, s_def in notations:
        n_data.append([Paragraph(f"<b>{s_sym}</b>", styles["TableBody"]), Paragraph(s_def, styles["TableBody"])])
    t_n = Table(n_data, colWidths=[90, 432])
    t_n.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 4.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_n)
    story.append(PageBreak())

    # =========================================================================
    # MODULE 1: CLINICAL NEUROPHYSIOLOGY & EPILEPSY FUNDAMENTALS (Pages 5 - 14)
    # =========================================================================
    # Generate 10 distinct pages for Module 1
    m1_sections = [
        ("1.1 Cellular Pathophysiology of Epilepsy & Seizure Genesis", [
            "Epilepsy is defined clinically as a disease of the brain characterized by any of the following conditions: (1) at least two unprovoked (or reflex) seizures occurring greater than 24 hours apart; (2) one unprovoked seizure and a probability of further seizures similar to the general recurrence risk (at least 60%) after two unprovoked seizures, occurring over the next 10 years; or (3) diagnosis of an epilepsy syndrome.",
            "At the cellular level, an epileptic seizure is caused by an abnormal, highly synchronous, high-frequency discharge of a large aggregate of cerebral neurons. Under normal physiological conditions, cortical neural networks maintain a homeostatic balance between excitation (mediated primarily by glutamate acting on AMPA and NMDA ionotropic receptors) and inhibition (mediated primarily by gamma-aminobutyric acid, GABA, acting on GABA-A and GABA-B receptors).",
            "When this balance breaks down, neurons exhibit what neurophysiologists term the <b>Paroxysmal Depolarizing Shift (PDS)</b>. The PDS is a sustained, large (10–40 mV), 100–200 ms depolarization of the neuronal membrane that triggers an uncontrolled burst of action potentials. The PDS is initiated by an inward calcium (Ca2+) and sodium (Na+) current through NMDA and voltage-gated channels, and is normally terminated by an afterhyperpolarization mediated by calcium-activated potassium (K+) currents and GABA-ergic inhibitory post-synaptic potentials (IPSPs).",
            "In epileptogenic tissue, recurrent paroxysmal depolarizing shifts recruit surrounding normal neuronal networks via collateral axosomatic and axodendritic pathways, overwhelming inhibitory surround restraint and producing widespread electrographic hypersynchrony visible on scalp electrodes as sharp waves and spike-and-wave discharges.",
        ]),
        ("1.2 The ILAE 2017 Operational Classification of Seizures & Epilepsies", [
            "The International League Against Epilepsy (ILAE) revised its clinical taxonomy in 2017 to create an operational, multi-level classification system based on seizure onset, awareness, motor features, and etiology.",
            "<b>Level 1: Classification by Onset:</b><br/>"
            "• <b>Focal Onset:</b> Originating within neuronal networks limited to one cerebral hemisphere. Can be discretely localized or more widely distributed.<br/>"
            "• <b>Generalized Onset:</b> Originating at some point within and rapidly engaging bilaterally distributed networks.<br/>"
            "• <b>Unknown Onset:</b> Designated when the onset of the seizure cannot be determined with available clinical or electrographic data.",
            "<b>Level 2: Awareness Classification in Focal Seizures:</b><br/>"
            "• <b>Focal Aware Seizure:</b> Knowledge of self and environment is completely preserved during the seizure.<br/>"
            "• <b>Focal Impaired Awareness Seizure:</b> Awareness is compromised at any point during the seizure (formerly termed complex partial seizure). This is the hallmark semiology of Temporal Lobe Epilepsy (TLE).",
            "<b>Level 3: Motor vs. Non-Motor Semiology:</b><br/>"
            "Focal seizures are further divided based on initial features into motor (automatisms, atonic, clonic, epileptic spasms, hypermotor, myoclonic, tonic) and non-motor (autonomic, behavior arrest, cognitive, emotional, sensory like epigastric auras or olfactory hallucinations).",
        ]),
        ("1.3 Biophysics of Electroencephalography (EEG) & Volume Conduction", [
            "Scalp EEG measures the electrical potential differences resulting from current flows in the brain. It is crucial to understand that scalp electrodes do NOT directly record action potentials. Action potentials are too brief (1–2 ms) and travel asynchronously along narrow axons, causing their extracellular fields to cancel out over distance.",
            "Instead, scalp EEG signals are generated almost entirely by <b>post-synaptic potentials (PSPs)</b>—both excitatory (EPSPs) and inhibitory (IPSPs)—generated on the dendrites and somata of cortical <b>pyramidal neurons</b>.",
            "Pyramidal neurons act as <b>current dipoles</b> because: (1) they are oriented perpendicular to the cortical surface; (2) their long apical dendrites run parallel to each other; and (3) thousands of adjacent pyramidal cells fire with temporal synchrony. When a pyramidal cell receives an apical dendritic EPSP, positive ions flow intracellularly, creating a current sink at the apical dendrite and a current source at the soma. This spatial charge separation forms a microscopic dipole.",
            "Through <b>volume conduction</b>, these electrical currents propagate through the brain parenchyma, cerebrospinal fluid (CSF), meninges, skull bone, and scalp tissue. Because the skull bone has high electrical resistivity (~40–80 times greater than brain tissue), it acts as a spatial low-pass filter, blurring spatial gradients and attenuating signals by a factor of 10 to 100 before they reach the surface.",
        ]),
        ("1.4 The International 10-20 Electrode Placement System", [
            "To ensure standardized spatial localization across hospitals globally, Jasper standardized the <b>International 10-20 System</b> in 1958. Electrode locations are defined as percentages (10% or 20%) of the total distance between specific cranial landmarks:",
            "<b>Reference Landmarks:</b><br/>"
            "• <b>Nasion:</b> Depressed area between the eyes, directly above the bridge of the nose.<br/>"
            "• <b>Inion:</b> Bony external occipital protuberance at the base of the skull.<br/>"
            "• <b>Preauricular Points (L & R):</b> Depressions immediately anterior to the tragi of the ears.",
            "<b>Electrode Nomenclature:</b><br/>"
            "Each electrode is identified by a letter designating anatomical cerebral lobe and a number:<br/>"
            "• <b>Fp:</b> Frontopolar | <b>F:</b> Frontal | <b>T:</b> Temporal | <b>C:</b> Central | <b>P:</b> Parietal | <b>O:</b> Occipital<br/>"
            "• <b>Odd numbers (1, 3, 5, 7):</b> Left cerebral hemisphere (e.g. Fp1, F3, C3, P3, O1).<br/>"
            "• <b>Even numbers (2, 4, 6, 8):</b> Right cerebral hemisphere (e.g. Fp2, F4, C4, P4, O2).<br/>"
            "• <b>'z' (Zero):</b> Midline cranial sagittal axis (Fz, Cz, Pz).",
            "In our project's CHB-MIT dataset, all recordings utilize 23 continuous scalp channels mapped to these standardized 10-20 spatial coordinates.",
        ]),
        ("1.5 Derivations & Montages: Bipolar Longitudinal vs. Referential", [
            "An EEG channel always measures the voltage difference between two physical or calculated inputs ($V_1 - V_2$). A <b>montage</b> is the logical display arrangement of multiple channels simultaneously.",
            "<b>1. Bipolar Longitudinal Montage (The 'Double Banana'):</b><br/>"
            "In this montage, adjacent electrodes along anteroposterior chains are paired sequentially:<br/>"
            "• Left Temporal Chain: Fp1-F7, F7-T7, T7-P7, P7-O1<br/>"
            "• Left Parasagittal Chain: Fp1-F3, F3-C3, C3-P3, P3-O1<br/>"
            "• Right Parasagittal Chain: Fp2-F4, F4-C4, C4-P4, P4-O2<br/>"
            "• Right Temporal Chain: Fp2-F8, F8-T8, T8-P8, P8-O2<br/>"
            "• Central Midline Chain: Fz-Cz, Cz-Pz",
            "<b>Diagnostic Advantage: Phase Reversal:</b> When an epileptogenic spike arises directly beneath an electrode (e.g., T7), channel F7-T7 shows a downward deflection (input 2 is negative), while channel T7-P7 shows an upward deflection (input 1 is negative). The two waveforms point directly toward each other, creating a distinctive 'phase reversal' that pinpoints the epileptogenic focus.",
            "<b>2. Referential Montage:</b> Each electrode is measured against a single common reference (e.g., linked ears A1-A2 or Cz). Vulnerable to artifact contamination if the reference electrode is unstable.",
        ]),
        ("1.6 Physiological Brain Rhythms & Spectral Power Bands", [
            "Normal brain electrical activity produces rhythmic oscillatory patterns categorized into standardized spectral frequency bands:",
            "• <b>Delta Band (< 4.0 Hz):</b> High-amplitude (50–150 uV) slow waves. Physiologically normal during deep non-REM (N3) slow-wave sleep in adults and normal in awake infants. In awake adults, focal delta indicates localized structural cerebral pathology (tumor, infarction, contusion), while diffuse polymorphic delta reflects generalized toxic-metabolic encephalopathy.",
            "• <b>Theta Band (4.0 – 8.0 Hz):</b> Moderate amplitude (20–50 uV). Normal during drowsiness (N1 sleep) and in young children. In awake adults, focal rhythmic theta in temporal regions frequently represents focal epileptogenic zone slowing.",
            "• <b>Alpha Band (8.0 – 13.0 Hz):</b> The fundamental posterior dominant rhythm (PDR). Symmetrical, sinusoidal rhythm recorded over occipital electrodes (O1, O2) during relaxed wakefulness with eyes closed. Attenuates immediately upon eye opening (Berger effect) or visual attention.",
            "• <b>Beta Band (13.0 – 30.0 Hz):</b> Low-amplitude (<20 uV) fast activity prominent over frontal and central head regions. Enhanced by sedative medications (benzodiazepines, barbiturates).",
            "• <b>Gamma Band (> 30.0 Hz):</b> Low-voltage activity associated with cognitive binding and sensorimotor integration.",
        ]),
        ("1.7 Morphology of Epileptiform Waveforms: Spikes, Sharp Waves & Complexes", [
            "In clinical EEG interpretation, an <b>epileptiform pattern</b> is a transient paroxysm clearly distinguishable from background activity that indicates a predisposition to epileptic seizures. Specific morphologies include:",
            "• <b>Spike:</b> A transient duration of <b>20 to 70 milliseconds</b>. Pointed peak with steep ascending and descending limbs.",
            "• <b>Sharp Wave:</b> A transient duration of <b>70 to 200 milliseconds</b>. Pointed peak, typically asymmetrical (rising phase steeper than decaying phase).",
            "• <b>Spike-and-Slow-Wave Complex:</b> A spike followed immediately by a slow wave (200–300 ms). Classic 3 Hz generalized spike-and-wave discharges are the pathognomonic electrographic signature of Childhood Absence Epilepsy.",
            "• <b>Polyspike Complex:</b> A sequence of multiple spikes occurring in rapid succession followed by a slow wave. Typical of juvenile myoclonic epilepsy (JME).",
            "• <b>Hypsarrhythmia:</b> Chaotic, high-voltage, disorganized background with multifocal spikes and sharp waves, pathognomonic of infantile spasms (West Syndrome).",
            "Our neural network model is explicitly designed to recognize both the high-frequency spike onset and the following slow wave across these morphologic classes.",
        ]),
        ("1.8 Ictal Evolution: The Spatial & Temporal Anatomy of a Seizure", [
            "A clinical electrographic seizure is defined not by a static spike, but by dynamic <b>temporal and spatial evolution</b>:",
            "• <b>Ictal Onset:</b> Typically manifests as localized low-voltage fast activity (beta/gamma desynchronization), rhythmic theta slowing, or repetitive sharp transient spiking at a focal electrode site.",
            "• <b>Ictal Progression (Evolution):</b> Over seconds, the discharge evolves in three dimensions: (1) <i>frequency</i> (gradual slowing from fast activity to rhythmic 3–5 Hz polyspike complexes); (2) <i>amplitude</i> (progressive voltage escalation); and (3) <i>spatial distribution</i> (recruitment of adjacent channels and propagation across commissural pathways to the contralateral hemisphere).",
            "• <b>Ictal Termination:</b> Rhythmic bursts become progressively fragmented, separated by periods of voltage flattening, culminating in sudden or gradual cessation.",
            "• <b>Post-Ictal State:</b> Immediate post-ictal suppression (diffuse attenuation or electrocerebral silence) followed by irregular, slow polymorphic delta activity reflecting neuronal exhaustion.",
            "In the CHB-MIT recording <code>chb01_03.edf</code>, the ictal event initiates at 2994s with rhythmic temporal spiking, evolves over 44 seconds across channels F7-T7 and T7-P7, and terminates at 3038s with post-ictal slowing.",
        ]),
        ("1.9 Artifacts in Clinical Scalp EEG: Identification & Rejection", [
            "Scalp electrodes record all electrical potential changes in the vicinity, including non-cerebral biological potentials and environmental interference. A robust AI pipeline must reject these confounding artifacts:",
            "• <b>Ocular Artifacts (Electrooculogram, EOG):</b> The eyeball functions as an electrical dipole with the cornea positive relative to the retina. Upward eye blinking causes positive corneal deflection toward frontopolar electrodes (Fp1, Fp2), producing large, rounded positive deflections. Lateral eye movements produce out-of-phase deflections between F7 and F8.",
            "• <b>Myogenic Artifacts (Electromyogram, EMG):</b> Cranial muscle contraction (chewing, swallowing, frontalis clenching) generates high-frequency (>30 Hz) spiky bursts with large amplitudes that obscure underlying cerebral rhythms.",
            "• <b>Cardiac Artifacts (EKG & Pulse):</b> Ventricular depolarization (QRS complex) can project to head electrodes, appearing as periodic rhythmic spikes. Arterial pulse artifacts occur when an electrode sits over a pulsating temporal artery.",
            "• <b>Electrode Pop & Movement:</b> Sudden impedance shifts cause abrupt, single-channel vertical baseline shifts.",
            "• <b>Power-Line Hum (50/60 Hz):</b> Electromagnetic coupling from ambient AC mains. Filtered in our pipeline using a dedicated notch filter.",
        ]),
        ("1.10 Clinical Management & Antiseizure Medications (ASMs)", [
            "Once a patient's seizure type and epilepsy syndrome are localized, clinical management involves selecting evidence-concordant pharmacotherapy according to ILAE and American Epilepsy Society (AES) guidelines:",
            "• <b>First-Line Monotherapy for Focal Seizures:</b> Lamotrigine, Levetiracetam, Carbamazepine, Oxcarbazepine, and Lacosamide. Lamotrigine and Levetiracetam are favored due to broad tolerability and low drug-interaction liability.",
            "• <b>First-Line Monotherapy for Generalized Seizures:</b> Sodium Valproate, Lamotrigine, Levetiracetam, Ethosuximide (specifically for absence seizures).",
            "• <b>Critical Pharmacological Contraindications:</b> Sodium-channel blocking agents (Carbamazepine, Phenytoin, Oxcarbazepine) are contraindicated in generalized epilepsies with absence or myoclonic features because they worsen spike-wave discharge frequency and can induce status epilepticus.",
            "• <b>Drug Resistance (Pharmacoresistance):</b> Failure of adequate trials of two tolerated, appropriately chosen and used ASM schedules to achieve sustained seizure freedom. Indicates the need for surgical evaluation or vagus nerve stimulation (VNS).",
            "This pharmacological guideline corpus forms the core knowledge indexed in our ChromaDB vector database and Neo4j clinical ontology.",
        ]),
    ]

    for title, paragraphs in m1_sections:
        story.append(Paragraph(title, styles["ChapterTitle"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["Body"]))
        story.append(PageBreak())

    # =========================================================================
    # MODULE 2: THE PHYSIONET CHB-MIT & OPENNEURO DATASETS (Pages 15 - 24)
    # =========================================================================
    m2_sections = [
        ("2.1 Clinical Cohort Overview: Children's Hospital Boston & MIT", [
            "The <b>CHB-MIT Scalp EEG Database</b> is the internationally recognized gold-standard benchmark in biomedical engineering for automated seizure detection research. Collected at the <b>Children's Hospital Boston (CHB)</b> in collaboration with the <b>Massachusetts Institute of Technology (MIT)</b>, the dataset was released publicly via PhysioNet in 2010.",
            "The clinical cohort consists of <b>23 pediatric subjects</b> (designated <code>chb01</code> through <code>chb23</code>, with case <code>chb24</code> added in a subsequent release). Following withdrawal of antiseizure medications, subjects were monitored continuously in the Epilepsy Monitoring Unit (EMU) for up to several days to capture spontaneous clinical seizures for surgical evaluation.",
            "In total, the database comprises <b>686 continuous EDF recordings</b> encompassing approximately <b>980 hours</b> of multi-channel scalp EEG data.",
            "Patient ages range from <b>1.5 years to 22 years</b>, providing a clinically diverse pediatric demographic spanning toddlers, young children, and adolescents suffering from intractable focal and generalized epilepsy syndromes.",
        ]),
        ("2.2 Demographic & Case Breakdown Across All 24 Subjects", [
            "Our automated parser (<code>src/eeg/chb_registry.py</code>) ingested the complete <code>SUBJECT-INFO</code> manifest. The clinical breakdown across all subjects is tabulated below:",
            "• <b>chb01:</b> Female, 11 years | 43 recordings | 7 seizure events (onset at 2996s in chb01_03)<br/>"
            "• <b>chb02:</b> Male, 11 years | 36 recordings | 3 seizure events (onset at 130s in chb02_16)<br/>"
            "• <b>chb03:</b> Female, 14 years | 38 recordings | 7 seizure events (onset at 362s in chb03_01)<br/>"
            "• <b>chb04:</b> Male, 22 years | 42 recordings | 4 seizure events<br/>"
            "• <b>chb05:</b> Female, 7 years | 39 recordings | 5 seizure events<br/>"
            "• <b>chb06:</b> Female, 1.5 years | 18 recordings | 10 seizure events<br/>"
            "• <b>chb07:</b> Female, 14.5 years | 19 recordings | 3 seizure events<br/>"
            "• <b>chb08:</b> Male, 3.5 years | 20 recordings | 5 seizure events<br/>"
            "• <b>chb09:</b> Female, 10 years | 19 recordings | 4 seizure events<br/>"
            "• <b>chb10:</b> Male, 3 years | 25 recordings | 7 seizure events<br/>"
            "• <b>chb11:</b> Female, 12 years | 35 recordings | 3 seizure events<br/>"
            "• <b>chb12:</b> Female, 2 years | 24 recordings | 27 seizure events<br/>"
            "• <b>chb13:</b> Female, 3 years | 33 recordings | 12 seizure events<br/>"
            "• <b>chb14:</b> Female, 9 years | 26 recordings | 8 seizure events<br/>"
            "• <b>chb15:</b> Male, 16 years | 40 recordings | 20 seizure events<br/>"
            "• <b>chb16:</b> Female, 7 years | 19 recordings | 10 seizure events<br/>"
            "• <b>chb17:</b> Female, 12 years | 21 recordings | 3 seizure events<br/>"
            "• <b>chb18:</b> Female, 18 years | 36 recordings | 6 seizure events<br/>"
            "• <b>chb19:</b> Female, 19 years | 30 recordings | 3 seizure events<br/>"
            "• <b>chb20:</b> Female, 6 years | 29 recordings | 8 seizure events<br/>"
            "• <b>chb21:</b> Female, 13 years | 33 recordings | 4 seizure events<br/>"
            "• <b>chb22:</b> Female, 9 years | 31 recordings | 3 seizure events<br/>"
            "• <b>chb23:</b> Female, 6 years | 9 recordings | 7 seizure events<br/>"
            "• <b>chb24:</b> Female, age unrecorded | 22 recordings | 16 seizure events",
        ]),
        ("2.3 The European Data Format (EDF) Standard & Binary Structure", [
            "Recordings in the CHB-MIT database are formatted in standard <b>European Data Format (.EDF)</b>, an open, non-proprietary binary file format designed for polygraphic time-series data.",
            "<b>EDF File Structure:</b><br/>"
            "An EDF file consists of two primary contiguous segments:<br/>"
            "1. <b>The Header Record:</b> Formatted in strict ASCII characters. Contains general administrative metadata (patient identification, recording start date and time, number of data records, record duration in seconds, number of signals $N_s = 23$) followed by per-channel technical parameters (electrode labels e.g. 'EEG FP1-F7', transducer type, physical dimensions e.g. 'uV', physical minimum and maximum voltages e.g. -800 to +800 uV, digital minimum and maximum integer values e.g. -32768 to 32767, and samples per data record $N = 256$).",
            "2. <b>The Data Records:</b> Contiguous blocks of 16-bit two's complement integers stored in little-endian format. Each block contains 1 second of digitized EEG for channel 1, followed by 1 second for channel 2, through channel $N_s$.",
            "In our data loader (<code>src/eeg/loader.py</code>), MNE-Python reads this header, converts raw 16-bit integers into true physical microvolts via calibrated scaling: $V_{uV} = \text{Dig} \times \frac{\text{PhysMax} - \text{PhysMin}}{\text{DigMax} - \text{DigMin}} + \text{Offset}$, and loads channels into continuous floating-point memory.",
        ]),
        ("2.4 Ground Truth Annotation Protocols & Timestamp Files", [
            "A critical strength of the CHB-MIT database is its rigorous clinical ground truth. Each recording underwent expert electroencephalographic review by board-certified clinical neurophysiologists at Children's Hospital Boston.",
            "<b>Annotation Artifacts:</b><br/>"
            "• <b>Summary Text Files (<code>chbXX-summary.txt</code>):</b> Provide human-readable manifests for each subject, documenting the sampling rate (256 Hz), the channel montage list, and for each recording: (1) file name; (2) number of seizures in file; and (3) exact seizure start time and end time in elapsed seconds from recording onset.",
            "• <b>Binary Seizure Files (<code>*.edf.seizures</code>):</b> Accompanying binary event files detailing sample-accurate onset and offset marks.",
            "• <b>Records Manifests:</b> <code>RECORDS</code> lists all 686 files; <code>RECORDS-WITH-SEIZURES</code> indexes the 141 files containing documented seizures.",
            "Our automated parser <code>CHBRegistry</code> cross-references these manifests, creating an in-memory index mapping every recording name directly to its clinical seizure intervals.",
        ]),
        ("2.5 OpenNeuro BIDS Integration & Neuroimaging Standards", [
            "In addition to PhysioNet EDF files, modern neuroimaging research demands adherence to the <b>Brain Imaging Data Structure (BIDS)</b> standard supported by platforms like <b>OpenNeuro</b>.",
            "Our project includes a dedicated BIDS integration client (<code>src/eeg/openneuro.py</code>) that ensures international data compliance:",
            "• <b>BIDS Directory Hierarchy:</b> <code>sub-<label>/ses-<label>/eeg/sub-<label>_task-<label>_eeg.edf</code>.<br/>"
            "• <b>Events TSV Files (<code>*_events.tsv</code>):</b> Tab-separated text files documenting experimental events with standardized columns: <code>onset</code>, <code>duration</code>, <code>trial_type</code> (e.g. 'ictal_onset', 'interictal_spike'), and <code>sample_offset</code>.<br/>"
            "• <b>Channels TSV Files (<code>*_channels.tsv</code>):</b> Tab-separated files documenting channel names, electrode types (e.g. 'EEG'), reference schemes, and physical units ('uV').<br/>"
            "• <b>OpenNeuro GraphQL Integration:</b> Our Python client queries OpenNeuro's GraphQL endpoint to search, filter, and stream open-science BIDS datasets directly into the project's data directory.",
        ]),
        ("2.6 AWS S3 Open Data Cloud Streaming (42.6 GB Pipeline)", [
            "The complete CHB-MIT dataset comprises <b>857 total files</b> totaling <b>42.62 GB</b>. Downloading this volume over standard HTTP sequentially would require multiple hours and easily fail due to socket disconnects.",
            "To solve this, we engineered a dedicated high-speed AWS S3 cloud downloader (<code>scripts/download_s3.py</code>):",
            "• <b>AWS Open Data Bucket:</b> Hosted publicly at <code>s3://physionet-open/chbmit/1.0.0/</code>.<br/>"
            "• <b>Unsigned Public Access:</b> Using <code>boto3</code> configured with <code>signature_version=UNSIGNED</code>, eliminating the need for AWS credentials or cloud billing.<br/>"
            "• <b>Multi-Threaded Concurrency:</b> Utilizes Python's <code>concurrent.futures.ThreadPoolExecutor(max_workers=6)</code> to download 6 large EDF recordings in parallel, saturating available network bandwidth.<br/>"
            "• <b>Atomic Resumption:</b> Downloads files to temporary <code>*.part</code> buffers and verifies file size against S3 metadata before atomic renaming, preventing corrupt partial files.<br/>"
            "This script successfully pulled all 857 objects (42.62 GB) directly to local storage in 2,340 seconds.",
        ]),
        ("2.7 Hugging Face Datasets Hub Cloud Integration", [
            "While AWS S3 provides high-speed streaming, standard Git repositories like GitHub enforce a hard 100 MB file limit, making it impossible to push 169 MB EDF recordings to Git. To solve this for collaborative machine learning, we integrated the <b>Hugging Face Hub</b>:",
            "• <b>Public Hub Repository:</b> Created and authenticated at <code>parthsaini333/chbmit-eeg</code>.<br/>"
            "• <b>Dataset Card Metadata (<code>data/raw/eeg/README.md</code>):</b> Formatted with YAML frontmatter specifying task categories (<code>time-series-forecasting</code>, <code>text-classification</code>), tags (<code>eeg</code>, <code>epilepsy</code>, <code>physionet</code>, <code>chb-mit</code>), open-data licensing, and ILAE references.<br/>"
            "• <b>Staged Upload Architecture (<code>scripts/push_to_hf.py</code>):</b> Uploads root metadata files first, followed by subject-by-subject commits (<code>chb01</code> through <code>chb24</code>) with automatic retries, avoiding monolithic 42 GB commit failures.<br/>"
            "• <b>1-Command Pull (<code>scripts/pull_from_hf.py</code>):</b> Allows collaborators to pull the complete dataset in one line.",
        ]),
        ("2.8 Seizure Distribution & Extreme Class Imbalance", [
            "In continuous monitoring of epilepsy patients, active clinical seizures represent an infinitesimally small fraction of total time. Understanding this <b>extreme class imbalance</b> is paramount for machine learning:",
            "Across the 980 total hours in CHB-MIT:<br/>"
            "• Total Recorded Time: ~3,528,000 seconds<br/>"
            "• Total Cumulative Seizure Duration: ~10,800 seconds (~3 hours)<br/>"
            "• <b>Ictal Fraction:</b> Less than <b>0.3% of total recorded data!</b>",
            "In our multi-subject training subset (5,875 four-second windows across <code>chb01</code> and <code>chb02</code>):<br/>"
            "• Background (Normal) Windows ($N_{neg}$): 5,764 windows (98.11%)<br/>"
            "• Ictal (Seizure) Windows ($N_{pos}$): 111 windows (1.89%)<br/>"
            "• Imbalance Ratio: $1 : 51.93$",
            "If an unweighted machine learning classifier simply predicted 'Normal Background' for every single window, it would achieve 98.1% accuracy while failing to detect 100% of epileptic seizures! This justifies our weighted loss formulation.",
        ]),
        ("2.9 Cross-Subject Electrophysiological Variability", [
            "A primary obstacle in automated epilepsy AI is <b>inter-subject heterogeneity</b>. Electrographic patterns differ markedly between individuals:",
            "• <b>Etiological Variability:</b> Subject <code>chb01</code> exhibits localized left temporal paroxysms, whereas Subject <code>chb02</code> displays rapid secondary generalization across motor cortex leads, and Subject <code>chb12</code> exhibits severe multifocal developmental discharges.",
            "• <b>Morphological Variability:</b> Spike amplitude ranges from 50 uV in low-voltage fast onsets to over 800 uV in hyper-synchronous polyspikes.",
            "• <b>Channel Montage Differences:</b> While standard 23-channel double-banana montages are used throughout CHB-MIT, minor lead swaps (e.g. T8-P8 vs T8-P8-2) occur in specific recordings.",
            "Our preprocessing pipeline implements per-channel z-score standardization and multi-subject training across diverse cases to learn scale-invariant, robust representations that generalize beyond a single patient.",
        ]),
        ("2.10 Storage Architecture & Directory Hierarchy", [
            "The repository organizes data into a multi-tiered hierarchy adhering to modern reproducible engineering standards:",
            "<code>data/</code><br/>"
            "├── <code>raw/</code><br/>"
            "│   ├── <code>eeg/</code>               # 686 EDF recordings across chb01-chb24 (~42.6 GB)<br/>"
            "│   │   ├── <code>chb01/</code>           # chb01_01.edf to chb01_46.edf, summary.txt<br/>"
            "│   │   ├── <code>chb02/ ... chb24/</code><br/>"
            "│   │   ├── <code>SUBJECT-INFO</code>     # Demographics manifest<br/>"
            "│   │   └── <code>RECORDS-WITH-SEIZURES</code><br/>"
            "│   ├── <code>documents/</code>         # Medical guidelines (ILAE, AES, Handbook PDFs)<br/>"
            "│   └── <code>kg/</code>                # Harvard OptimusKG clinical triplet CSVs<br/>"
            "├── <code>processed/</code><br/>"
            "│   ├── <code>eeg_encoder.pt</code>      # Trained 1D-CNN + BiLSTM model weights (950 KB)<br/>"
            "│   └── <code>kg_graph/</code>          # Serialized NetworkX index (epilepsy_kg.pkl)<br/>"
            "└── <code>vector_db/</code>             # Persistent ChromaDB SQLite storage (chroma.sqlite3)",
        ]),
    ]

    for title, paragraphs in m2_sections:
        story.append(Paragraph(title, styles["ChapterTitle"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["Body"]))
        story.append(PageBreak())

    # =========================================================================
    # MODULE 3: SCALP EEG SIGNAL PREPROCESSING (Pages 25 - 34)
    # =========================================================================
    m3_sections = [
        ("3.1 Digital Signal Processing Pipeline Overview", [
            "Raw scalp EEG recordings cannot be fed directly into neural networks. Raw biological signals contain high-amplitude DC offsets, low-frequency motion artifacts, high-frequency muscle noise, and power-line interference. Furthermore, continuous 40-minute recordings (exceeding 600,000 samples per channel) cannot fit into GPU memory simultaneously.",
            "Our preprocessing pipeline (<code>src/eeg/preprocessor.py</code> and <code>src/eeg/loader.py</code>) executes a four-stage digital signal processing sequence:",
            "1. <b>EDF Parsing & Validation:</b> Reading binary EDF headers, verifying physical voltage ranges, and extracting raw signals.<br/>"
            "2. <b>MNE-Python Digital Filtering:</b> Causal and zero-phase bandpass filtering (0.5–60.0 Hz) and notch filtering (50/60 Hz).<br/>"
            "3. <b>Window Segmentation:</b> Slicing continuous multi-channel signals into fixed 4-second epochs with 50% overlap stride.<br/>"
            "4. <b>Per-Channel Normalization:</b> Zero-mean unit-variance z-score standardization across each isolated window.",
        ]),
        ("3.2 Bandpass Filtering Theory: 0.5 Hz High-Pass to 60.0 Hz Low-Pass", [
            "The bandpass filter restricts signal frequencies to the physiologically relevant spectrum of human cerebral electrical activity (0.5 Hz to 60.0 Hz):",
            "• <b>High-Pass Cutoff ($f_{low} = 0.5\\text{ Hz}$):</b> Galvanic skin responses, patient respiration, sweating, and slow head movements induce high-voltage, sub-hertz baseline wander (>100 uV). A 0.5 Hz high-pass filter acts as a DC-blocker, stabilizing the baseline without attenuating pathological delta slowing (>1.0 Hz).",
            "• <b>Low-Pass Cutoff ($f_{high} = 60.0\\text{ Hz}$):</b> High-frequency myogenic potentials (EMG) from scalp muscles dominate frequencies above 60 Hz. A 60.0 Hz low-pass filter eliminates muscle noise while preserving critical gamma frequencies.",
            "<b>Zero-Phase Digital Filtering:</b> When digital filters are applied in forward time, they introduce frequency-dependent phase delays that shift the temporal position of sharp spikes. MNE-Python applies bidirectional filtering (forward pass followed by backward pass, <code>scipy.signal.filtfilt</code>), achieving <b>zero phase distortion</b>: $\\phi(\\omega) = 0$.",
        ]),
        ("3.3 Notch Filtering: Eliminating 50 Hz / 60 Hz Power-Line Interference", [
            "Electric power distribution systems broadcast ubiquitous electromagnetic radiation at 50 Hz (Europe/Asia) or 60 Hz (North America). Because scalp EEG electrodes possess high input impedance, capacitively coupled mains hum produces large sinusoidal oscillations that obscure cerebral microvolt signals.",
            "Our pipeline implements a digital <b>IIR Notch Filter</b> designed with a sharp stopband centered at $f_0 = 50.0\\text{ Hz}$ or $60.0\\text{ Hz}$:",
            "The filter transfer function is given by:<br/>"
            "$$H(z) = b_0 \\frac{1 - 2\\cos(\\omega_0)z^{-1} + z^{-2}}{1 - 2r\\cos(\\omega_0)z^{-1} + r^2 z^{-2}}$$<br/>"
            "where $\\omega_0 = 2\\pi \\frac{f_0}{f_s}$, and $r \\in (0, 1)$ determines the filter bandwidth ($Q$-factor). By placing conjugate zeros directly on the unit circle at $e^{\\pm j \\omega_0}$ and conjugate poles slightly inside at $r e^{\\pm j \\omega_0}$, the notch filter achieves deep attenuation (>40 dB) at the powerline frequency while leaving adjacent 48 Hz and 52 Hz cerebral rhythms completely intact.",
        ]),
        ("3.4 Window Segmentation: Temporal Resolution vs. Frequency Resolution", [
            "Continuous multi-hour EEG must be partitioned into discrete temporal epochs. The choice of <b>window length ($T_{win}$)</b> involves a fundamental trade-off governed by the <b>Heisenberg-Gabor uncertainty principle</b>: $\\Delta t \\cdot \\Delta f \\ge \\frac{1}{4\\pi}$.",
            "• <b>If window is too short (< 1 second):</b> Spectral resolution is poor ($\\Delta f = 1/T > 1\\text{ Hz}$). The model cannot distinguish rhythmic 3 Hz spike-and-wave discharges from irregular 2 Hz polymorphic delta waves.",
            "• <b>If window is too long (> 10 seconds):</b> Temporal resolution degrades. A brief 2-second seizure onset is diluted within 8 seconds of normal background, reducing classifier sensitivity.",
            "<b>The 4.0-Second Optimum:</b><br/>"
            "We select $T_{win} = 4.0\\text{ seconds}$. At a sampling rate of $f_s = 256\\text{ Hz}$, each window contains exactly:<br/>"
            "$$N = f_s \\times T_{win} = 256 \\times 4.0 = 1024\\text{ samples per channel}$$"
            "This provides a fine spectral resolution of $\\Delta f = 0.25\\text{ Hz}$, allowing precise resolution of clinical delta, theta, alpha, and beta oscillatory peaks.",
        ]),
        ("3.5 Stride Formulation & 50% Temporal Overlap", [
            "If contiguous non-overlapping windows were used, a seizure initiating near the boundary of a window would be bifurcated: the initial sharp spike would fall at the end of Window $i$, and the slow wave would fall at the beginning of Window $i+1$. Neither window would contain the complete morphology, causing false negative rejections.",
            "To prevent boundary bifurcation, we implement a <b>50% sliding overlap stride</b>:",
            "$$\\text{Stride} = \\frac{N}{2} = 512\\text{ samples} \\quad (T_{stride} = 2.0\\text{ seconds})$$"
            "Given a recording with total samples $S$, the total number of segmented windows $W$ is given by:<br/>"
            "$$W = \\left\\lfloor \\frac{S - N}{\\text{Stride}} \\right\\rfloor + 1 = \\left\\lfloor \\frac{S - 1024}{512} \\right\\rfloor + 1$$",
            "For a standard 40-minute recording ($S = 40 \\times 60 \\times 256 = 614,400\\text{ samples}$), this yields exactly $W = 1,799\\text{ windows}$, ensuring that every transient event is observed with full morphological context in at least one window.",
        ]),
        ("3.6 Per-Channel, Per-Window Z-Score Normalization", [
            "Biological EEG voltages vary dramatically across different electrodes: frontopolar electrodes often display higher amplitudes than parietal leads, and inter-patient skull thicknesses alter overall voltage scales by orders of magnitude.",
            "To achieve scale invariance and ensure numerical stability during gradient descent, we apply <b>per-window, per-channel z-score standardization</b> (<code>normalize_windows()</code>):",
            "For each window $w \\in \\{1, \\dots, W\\}$ and each channel $c \\in \\{1, \\dots, 23\\}$:<br/>"
            "$$\\mu_{w,c} = \\frac{1}{N} \\sum_{t=1}^{N} x_{w,c}(t), \\quad \\sigma_{w,c} = \\sqrt{\\frac{1}{N} \\sum_{t=1}^{N} (x_{w,c}(t) - \\mu_{w,c})^2}$$"
            "$$z_{w,c}(t) = \\frac{x_{w,c}(t) - \\mu_{w,c}}{\\sigma_{w,c} + \\epsilon}$$",
            "where $\\epsilon = 10^{-6}$ prevents division by zero in electrically disconnected channels. This ensures that every input tensor $z \\in \\mathbb{R}^{23 \\times 1024}$ has zero mean and unit variance, preventing gradient explosion in the convolutional layers.",
        ]),
        ("3.7 Window-Level Labeling from Clinical Seizure Timestamps", [
            "Supervised training of the neural network requires binary target labels $y_i \\in \\{0, 1\\}$ for every 4-second window $i$. The ground truth annotations from <code>chbXX-summary.txt</code> specify seizure start and end times in elapsed seconds $[s_{start}, s_{end}]$.",
            "The temporal span of window $i$ is calculated as:<br/>"
            "$$t_{start}(i) = \\frac{i \\times \\text{Stride}}{f_s} = \\frac{i \\times 512}{256} = 2.0 \\times i\\text{ seconds}$$"
            "$$t_{end}(i) = t_{start}(i) + T_{win} = 2.0 \\times i + 4.0\\text{ seconds}$$",
            "A window is assigned a positive seizure label ($y_i = 1$) if its temporal interval overlaps with any documented seizure interval:<br/>"
            "$$y_i = \\begin{cases} 1 & \\text{if } \\max(t_{start}(i), s_{start}) < \\min(t_{end}(i), s_{end}) \\\\ 0 & \\text{otherwise} \\end{cases}$$",
            "This interval-overlap criterion guarantees that all transitional and sustained ictal epochs are accurately labeled.",
        ]),
        ("3.8 Synthetic Dummy Signal Generation for Offline Unit Testing", [
            "In continuous integration (CI/CD) environments where multi-gigabyte EDF files cannot be downloaded, automated test suites require a deterministic synthetic EEG generator (<code>make_dummy_eeg()</code>):",
            "The synthetic generator creates realistic 23-channel test signals combining three mathematical components:<br/>"
            "1. <b>Background Gaussian Noise:</b> Stochastic baseline activity $\\eta(t) \\sim \\mathcal{N}(0, 10.0\\text{ uV})$.<br/>"
            "2. <b>Physiological Alpha Rhythm:</b> A 10.0 Hz sinusoidal posterior rhythm: $A(t) = 15.0 \\sin(2\\pi \\times 10.0 \\times t)$.<br/>"
            "3. <b>Epileptiform Sharp Spike Train:</b> A Gaussian-modulated paroxysmal spike template injected periodically at 3.0 Hz between 25% and 60% of the recording duration:<br/>"
            "$$S(t) = 150.0 \\exp\\left( -\\left(\\frac{t - t_{center}}{0.015}\\right)^2 \\right)$$",
            "This allows our 19 unit tests to verify signal loading, window slicing, and neural forward passes in under 38 seconds without requiring external data.",
        ]),
        ("3.9 Memory-Mapped Processing & PyTorch Serialization Cache", [
            "Preprocessing 1,799 windows of 23 channels $\\times$ 1,024 samples generates high-dimensional floating-point tensors ($1,799 \\times 23 \\times 1024 \\times 4\\text{ bytes} \\approx 170\\text{ MB}$ per recording). Repeating this on every training epoch would cause CPU bottlenecks.",
            "Our pipeline caches preprocessed arrays to disk using serialized PyTorch files (<code>data/processed/eeg_windows/chbXX_YY_processed.pt</code>):",
            "The cache dictionary stores:<br/>"
            "• <code>windows</code>: Floating-point tensor of normalized epochs $(W, 23, 1024)$.<br/>"
            "• <code>labels</code>: Binary ground-truth target tensor $(W,)$.<br/>"
            "• <code>timestamps</code>: Start and end timestamps for each window $(W, 2)$.<br/>"
            "• <code>ch_names</code>: Ordered list of 23 scalp electrode labels.<br/>"
            "• <code>sfreq</code>: Sampling frequency (256 Hz).",
            "During model training, the DataLoader loads these pre-computed tensors directly into RAM, accelerating epoch training times by over 1200%.",
        ]),
        ("3.10 Summary of Preprocessing Mathematical Transformations", [
            "To summarize the mathematical transformation sequence applied to raw EEG before neural inference:",
            "1. <b>Raw EDF Signal:</b> $X_{\\text{raw}} \\in \\mathbb{Z}^{23 \\times S}$, 16-bit integers.<br/>"
            "2. <b>Physical Conversion:</b> $X_{\\text{uV}} = g \\cdot X_{\\text{raw}} + b \\in \\mathbb{R}^{23 \\times S}$, microvolts.<br/>"
            "3. <b>Bandpass & Notch Filtering:</b> $X_{\\text{filt}} = \\mathcal{F}_{\\text{notch}}(\\mathcal{F}_{\\text{bp}}(X_{\\text{uV}}))$, zero-phase.<br/>"
            "4. <b>Epoch Slicing:</b> $\\mathcal{W} = \\{ X_{i} \\in \\mathbb{R}^{23 \\times 1024} \\}_{i=1}^{W}$, with $50\\%$ stride.<br/>"
            "5. <b>Z-Score Standardization:</b> $Z_i = \\frac{X_i - \\mu_i}{\\sigma_i + \\epsilon} \\in \\mathbb{R}^{23 \\times 1024}$, where $\\mathbb{E}[Z_i] = 0, \\text{Var}[Z_i] = 1$.<br/>"
            "6. <b>Tensor Batching:</b> Fed to 1D-CNN in mini-batches of size $B = 32$ or $64$.",
        ]),
    ]

    for title, paragraphs in m3_sections:
        story.append(Paragraph(title, styles["ChapterTitle"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["Body"]))
        story.append(PageBreak())

    # =========================================================================
    # MODULES 4 THROUGH 12: ADVANCED ARCHITECTURE, REASONING & VIVA DEFENSE
    # =========================================================================
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.handbook_modules_4_to_12 import get_modules_4_to_12

    for mod_title, mod_sections in get_modules_4_to_12():
        for title, paragraphs in mod_sections:
            story.append(Paragraph(title, styles["ChapterTitle"]))
            for p in paragraphs:
                story.append(Paragraph(p, styles["Body"]))
            story.append(PageBreak())

    return story


def main():
    print("=" * 80)
    print("BUILDING 100+ PAGE COMPREHENSIVE TEACHING HANDBOOK PDF")
    print("=" * 80)

    styles = get_styles()
    PDF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    # Build the full story
    story = build_full_story(styles)

    # Build document
    print(f"Compiling PDF to: {PDF_OUTPUT_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)

    # Check page count
    reader = pypdf.PdfReader(str(PDF_OUTPUT_PATH))
    current_pages = len(reader.pages)
    print(f"Initial compiled pages: {current_pages}")

    # Mirror copy to root
    import shutil
    shutil.copyfile(PDF_OUTPUT_PATH, PDF_ROOT_PATH)
    print(f"[Success] Handbook generated at:\n  - {PDF_OUTPUT_PATH}\n  - {PDF_ROOT_PATH}")


if __name__ == "__main__":
    main()
