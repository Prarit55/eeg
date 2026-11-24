"""
Script to generate the Comprehensive Teaching Handbook & Viva Defense Guide PDF
for the Minor Project: Query-Adaptive Multimodal Self-RAG for Epilepsy Analysis.
Author: Parth Saini (06219051923), USAR, GGSIPU.
"""
from __future__ import annotations

import os
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

PDF_DEST = Path(__file__).resolve().parent.parent / "docs" / "EEG_Self_RAG_Comprehensive_Teaching_Handbook.pdf"
PDF_ROOT = Path(__file__).resolve().parent.parent / "EEG_Self_RAG_Comprehensive_Teaching_Handbook.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display total page count."""

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
        self.setFillColor(colors.HexColor("#718096"))

        # Running Header
        self.drawString(
            54, 750, "Query-Adaptive Multimodal Self-RAG for Epilepsy Analysis — Teaching & Viva Handbook"
        )
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Candidate: Parth Saini (06219051923) | USAR, GGSIPU | Course: ARP-455")
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_handbook():
    PDF_DEST.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_DEST),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_dark = colors.HexColor("#2D3748")      # Dark Gray Text
    c_light_bg = colors.HexColor("#EDF2F7")  # Very Light Gray Box
    c_accent = colors.HexColor("#0D9488")    # Teal Accent

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=c_primary,
        alignment=1,  # Center
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=24,
    )

    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=c_dark,
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "ChapterHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A202C"),
    )

    viva_q_style = ParagraphStyle(
        "VivaQuestion",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#991B1B"),  # Reddish
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    viva_a_style = ParagraphStyle(
        "VivaAnswer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=8,
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("TEACHING & VIVA DEFENSE HANDBOOK", title_style))
    story.append(
        Paragraph(
            "Complete Conceptual Mastery, Technical Architecture & Examiner Q&A Guide<br/>"
            "for <b>Query-Adaptive Multimodal Self-RAG in Epilepsy Analysis</b>",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=10, spaceAfter=20))

    meta_text = """
    <b>Author / Student:</b> Parth Saini (Roll No: <b>06219051923</b>)<br/>
    <b>Course:</b> Minor Project [ARP-455] — B.Tech Computer Science / Robotics<br/>
    <b>Supervisor:</b> Dr. Ruchika Lalit, Assistant Professor<br/>
    <b>Institution:</b> University School of Automation & Robotics (USAR)<br/>
    <b>Affiliation:</b> Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India<br/>
    <b>Live Web Application:</b> https://prarit55.github.io/eeg/<br/>
    <b>Source Code Repository:</b> https://github.com/Prarit55/eeg
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 30))

    # Abstract Box
    summary_box_data = [[
        Paragraph(
            "<b>Handbook Objective:</b> This document provides an exhaustive, conceptual, and mathematical breakdown "
            "of every technical component implemented in the minor project. It is written specifically for the student "
            "to study, internalize, and speak with confidence during the viva voce examination and project defense. "
            "Includes word-for-word model answers to tough examiner questions (e.g., BiLSTM data leakage, GraphRAG rationale, "
            "Self-RAG reflection mechanics, and full CHB-MIT dataset engineering).",
            callout_style,
        )
    ]]
    t_sum = Table(summary_box_data, colWidths=[500])
    t_sum.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#86EFAC")),
        ("PADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_sum)
    story.append(PageBreak())

    # =========================================================================
    # SECTION 1: THE 60-SECOND PRESENTATION PITCH
    # =========================================================================
    story.append(Paragraph("1. The 60-Second Presentation Elevator Pitch", h1_style))
    story.append(Paragraph(
        "<i>Memorize and adapt this exact opening speech when introducing your project to the viva panel:</i>",
        body_style,
    ))

    pitch_data = [[
        Paragraph(
            "\"Good morning, respected examiners. My minor project is titled <b>'Query-Adaptive Multimodal Self-RAG for Evidence-Grounded Medical Reasoning in Epilepsy Analysis'</b>.<br/><br/>"
            "In clinical epilepsy care, doctors review multi-hour 23-channel scalp EEG recordings alongside complex clinical history and ILAE medical guidelines. "
            "Today's AI systems face two critical failures: <b>first</b>, traditional ML models only perform binary seizure detection without any clinical reasoning; "
            "and <b>second</b>, standard LLMs hallucinate medical advice and blindly query every database, wasting compute on irrelevant evidence.<br/><br/>"
            "To solve this, I designed a <b>5-layer multimodal decision-support framework</b> that: "
            "(1) processes genuine scalp EEG signals using a 1D-CNN + BiLSTM neural encoder; "
            "(2) integrates dual knowledge channels—a <b>ChromaDB vector store</b> with dense medical embeddings and a <b>Neo4j Knowledge Graph</b> with 7,920 clinical relations; "
            "(3) dynamically scores query suitability to eliminate <b>33.3% of redundant retrieval queries</b>; "
            "and (4) enforces <b>Self-RAG reflection</b> to verify facts and eliminate medical hallucinations by <b>78.6%</b>.<br/><br/>"
            "I have trained and verified this across the entire <b>42 GB PhysioNet CHB-MIT database</b> (all 24 subjects, 686 recordings) and deployed a live clinical dashboard on GitHub Pages.\"",
            callout_style,
        )
    ]]
    t_pitch = Table(pitch_data, colWidths=[500])
    t_pitch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_pitch)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 2: CLINICAL BACKGROUND & CORE CONCEPTS
    # =========================================================================
    story.append(Paragraph("2. Clinical Background & Electrophysiology Fundamentals", h1_style))
    story.append(Paragraph(
        "To defend this project, you must understand the medical terminology as thoroughly as the machine learning algorithms:",
        body_style,
    ))

    story.append(Paragraph("A. What is Epilepsy & Seizure Classification?", h2_style))
    story.append(Paragraph(
        "<b>Epilepsy</b> is a chronic neurological disorder characterized by recurrent, unprovoked seizures caused by abnormal, "
        "excessive, synchronous electrical discharges of cerebral neurons. According to the <b>ILAE 2017 Operational Classification</b>, seizures are categorized by:",
        body_style,
    ))
    story.append(Paragraph("• <b>Focal Onset:</b> Originates within networks limited to one cerebral hemisphere. Can be <i>Focal Aware</i> (patient retains consciousness) or <i>Focal Impaired Awareness</i> (altered sensorium, common in Temporal Lobe Epilepsy).", bullet_style))
    story.append(Paragraph("• <b>Generalized Onset:</b> Originates at some point and rapidly engages bilaterally distributed networks (e.g., Absence seizures, Generalized Tonic-Clonic).", bullet_style))
    story.append(Paragraph("• <b>Unknown Onset:</b> When onset cannot be determined due to missing initial clinical or EEG evidence.", bullet_style))

    story.append(Paragraph("B. Scalp EEG Signals & The International 10-20 System", h2_style))
    story.append(Paragraph(
        "<b>Electroencephalography (EEG)</b> measures post-synaptic dendritic potentials from cortical pyramidal cells. "
        "In our project, we use the gold-standard <b>PhysioNet CHB-MIT database</b>:",
        body_style,
    ))
    story.append(Paragraph("• <b>Montage:</b> 23 bipolar longitudinal channels (the 'Double Banana' montage, e.g., FP1-F7, F7-T7, T7-P7, P7-O1). Bipolar montages measure voltage differences between adjacent electrode pairs, eliminating common-mode environmental noise.", bullet_style))
    story.append(Paragraph("• <b>Sampling Frequency:</b> 256 Hz (256 samples per second per channel, fulfilling the Nyquist criterion for physiological brain rhythms up to 128 Hz).", bullet_style))
    story.append(Paragraph("• <b>Clinical EEG Rhythms:</b> Delta (<4 Hz, slow waves), Theta (4–8 Hz, drowsy/focal slowing), Alpha (8–13 Hz, relaxed posterior rhythm), Beta (13–30 Hz, active cortex), Gamma (>30 Hz, cognitive binding).", bullet_style))
    story.append(Paragraph("• <b>Ictal vs. Interictal Activity:</b> <i>Ictal</i> refers to the active electrographic seizure event (characterized by sudden rhythmic, high-amplitude spikes, polyspikes, and slow-wave evolution). <i>Interictal</i> refers to the background activity between seizures.", bullet_style))

    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: SYSTEM ARCHITECTURE & 5 CORE LAYERS
    # =========================================================================
    story.append(Paragraph("3. System Architecture: The 5-Layer Reasoning Pipeline", h1_style))
    story.append(Paragraph(
        "Explain to examiners that your system is designed around 5 loosely-coupled, highly-modular layers:",
        body_style,
    ))

    layers_data = [
        [Paragraph("<b>Layer</b>", body_style), Paragraph("<b>Components & Code Files</b>", body_style), Paragraph("<b>Function & Innovation</b>", body_style)],
        [
            Paragraph("<b>Layer 1: Scalp EEG Neural Processing</b>", body_style),
            Paragraph("<code>src/eeg/loader.py</code><br/><code>src/eeg/models.py</code><br/><code>src/eeg/pipeline.py</code>", body_style),
            Paragraph("Filters raw EDF (0.5–60Hz bandpass + notch), segments into 4s windows (1024 samples, 50% stride). 1D-CNN extracts local temporal filters; BiLSTM captures bilateral spike-wave morphology. Outputs 128-d patient vector and detects seizure start/end times.", body_style)
        ],
        [
            Paragraph("<b>Layer 2: Dual Knowledge Retrieval</b>", body_style),
            Paragraph("<code>src/knowledge/vector_store.py</code><br/><code>src/knowledge/neo4j_adapter.py</code>", body_style),
            Paragraph("<b>Dual Retrieval:</b><br/>• <i>ChromaDB:</i> Dense semantic embeddings (BAAI/bge-small-en-v1.5) over ILAE/AES clinical guidelines.<br/>• <i>Neo4j / OptimusKG:</i> 2,571 clinical nodes & 7,920 triples for multi-hop relationship traversal (syndromes, lobes, ASMs).", body_style)
        ],
        [
            Paragraph("<b>Layer 3: Query Understanding & Routing</b>", body_style),
            Paragraph("<code>src/routing/classifier.py</code><br/><code>src/routing/decomposer.py</code><br/><code>src/routing/scorer.py</code>", body_style),
            Paragraph("Classifies user intent (definitional, relational, EEG-specific, clinical, complex). Decomposes complex queries. Calculates suitability scores S_vec, S_graph, S_eeg vs. threshold tau=0.5. <b>Bypasses irrelevant channels, reducing redundant calls by 33.3%.</b>", body_style)
        ],
        [
            Paragraph("<b>Layer 4: Multimodal Evidence Fusion</b>", body_style),
            Paragraph("<code>src/fusion/aggregator.py</code><br/><code>src/fusion/context_formatter.py</code>", body_style),
            Paragraph("Aggregates temporal electrographic intervals, patient EHR context, graph triples, and guideline passages into a structured prompt with granular citation handles ([Doc-i], [KG-Triplets], [EEG-Evidence]).", body_style)
        ],
        [
            Paragraph("<b>Layer 5: Self-RAG Iterative Refinement</b>", body_style),
            Paragraph("<code>src/self_rag/critique.py</code><br/><code>src/self_rag/generator.py</code><br/><code>src/self_rag/loop.py</code>", body_style),
            Paragraph("Active self-reflection evaluating [IsRel] (Relevance), [IsSup] (Support), [IsUse] (Utility). If critique score <0.70, it dynamically adapts tau, expands graph hop radius, and re-retrieves, <b>reducing medical hallucinations by 78.6%</b>.", body_style)
        ],
    ]
    t_layers = Table(layers_data, colWidths=[110, 130, 260])
    t_layers.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_layers)
    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: THE BIG VIVA QUESTIONS & DEFENSE REPERTOIRE
    # =========================================================================
    story.append(Paragraph("4. Tough Viva Questions & Word-for-Word Model Answers", h1_style))
    story.append(Paragraph(
        "Examiners will challenge you on architectural choices. Here are the exact questions they will ask and how to answer them:",
        body_style,
    ))

    # Q1: BiLSTM Data Leakage
    story.append(Paragraph("Q1: \"Why did you use a Bidirectional LSTM (BiLSTM) for EEG time-series? Doesn't a bidirectional model look into the future and cause temporal data leakage?\"", viva_q_style))
    q1_ans = """
    <b>Model Answer to Speak:</b><br/>
    \"Respected Examiner, that is an essential consideration in time-series modeling, but in our architecture, <b>the BiLSTM does NOT cause data leakage</b> for three distinct reasons:<br/>
    <b>1. Within-Window vs. Across-Timeline Isolation:</b> The BiLSTM operates strictly <i>inside</i> an isolated 4-second window (1,024 samples downsampled to 128 temporal steps by the 1D-CNN). The LSTM's hidden state is re-initialized for every window. It never accesses future minutes or future recording windows.<br/>
    <b>2. Clinical Biphasic Spike-Wave Morphology:</b> Epileptic paroxysms are not point impulses; they have a biphasic morphology consisting of a leading sharp depolarization spike (30–70 ms) followed by a trailing slow-wave hyperpolarization (200–300 ms). A forward-only LSTM only sees the spike and often misidentifies muscle artifacts as seizures. The backward LSTM pass confirms whether the trailing slow wave exists, enabling true morphological recognition.<br/>
    <b>3. Retrospective Diagnostic Task:</b> In clinical decision support, the doctor reviews an existing 40-minute EEG recording or a 4-second buffered epoch. This is retrospective clinical reasoning, where the full 4-second buffer is already digitized in memory with an acceptable 4s clinical latency.\"
    """
    story.append(Paragraph(q1_ans, viva_a_style))

    # Q2: Vector DB vs Knowledge Graph
    story.append(Paragraph("Q2: \"Why did you integrate BOTH a Vector Database (ChromaDB) and a Knowledge Graph (Neo4j)? Isn't a Vector Store sufficient for RAG?\"", viva_q_style))
    q2_ans = """
    <b>Model Answer to Speak:</b><br/>
    \"Vector databases and Knowledge Graphs have fundamentally complementary strengths and failure modes in medicine:<br/>
    <b>• Vector Store (ChromaDB):</b> Excels at unstructured semantic matching over clinical practice guidelines (e.g., retrieving descriptive treatment paragraphs and dosing schedules from ILAE and AES PDFs). However, vector embeddings suffer from <i>associative hallucinations</i> and cannot perform multi-hop relational logic.<br/>
    <b>• Knowledge Graph (Harvard OptimusKG & Neo4j):</b> Contains 2,571 clinical nodes and 7,920 validated relationships. It represents deterministic, multi-hop symbolic paths (e.g., <i>Temporal Lobe — [associated_with] — Focal Impaired Awareness Seizures — [treated_by] — Carbamazepine</i>). Cypher queries allow exact graph traversals that guarantee factual correctness.<br/>
    By fusing unstructured text chunks with structured knowledge triples, the framework eliminates semantic hallucinations while retaining detailed guideline explanations.\"
    """
    story.append(Paragraph(q2_ans, viva_a_style))

    # Q3: Query-Adaptive Routing & 33.3% Redundancy
    story.append(Paragraph("Q3: \"What is Query-Adaptive Dynamic Routing and how did you prove a 33.3% reduction in redundant module calls?\"", viva_q_style))
    q3_ans = """
    <b>Model Answer to Speak:</b><br/>
    \"In naive RAG frameworks, every user question triggers every retrieval channel simultaneously—vector search, graph querying, and EEG neural inference—regardless of whether the question actually needs them. For example, asking <i>'What is the definition of focal epilepsy?'</i> has zero need for GPU EEG signal processing.<br/>
    Our <b>QueryRouter</b> classifies user intent and decomposes complex questions into atomic subqueries. It then evaluates a continuous suitability scoring function $S_{vec}, S_{graph}, S_{eeg} \\in [0, 1]$ against a calibrated threshold $\\tau = 0.50$. A channel is activated if and only if $S_m(q) \\ge \\tau$.<br/>
    In our automated benchmark (<code>evaluation/eval_retrieval.py</code> across curated clinical test cases), standard RAG executed an average of 3.00 module calls per query, whereas our adaptive router reduced this to <b>2.00 calls per query</b>—achieving an exact <b>33.3% reduction in redundant computational calls</b> with 100% routing accuracy.\"
    """
    story.append(Paragraph(q3_ans, viva_a_style))

    # Q4: Self-RAG vs Standard RAG
    story.append(Paragraph("Q4: \"How does Self-RAG work and how does it prevent medical hallucinations?\"", viva_q_style))
    q4_ans = """
    <b>Model Answer to Speak:</b><br/>
    \"Standard RAG is a single-pass, feed-forward process: retrieve once and generate once. If retrieved passages are noisy or irrelevant, the LLM hallucinates an unsupported answer.<br/>
    <b>Self-RAG (Self-Reflective RAG)</b> introduces multi-token active critique:
    (1) <b>[IsRel] (Relevance):</b> Evaluates whether the retrieved evidence directly answers the patient's query.
    (2) <b>[IsSup] (Support):</b> Verifies sentence-by-sentence whether every claim in the response is strictly grounded in the retrieved citations.
    (3) <b>[IsUse] (Utility):</b> Scores the clinical usefulness and clarity of the output.<br/>
    In our <code>SelfRAGLoop</code> (<code>src/self_rag/loop.py</code>), if the critique score falls below 0.70, the system triggers an <b>active refinement loop</b>: it lowers the threshold $\\tau$, expands the graph hop radius to $k=2$, and re-retrieves new evidence before re-generating. This achieved a verified <b>78.6% reduction in unsupported medical statements</b>.\"
    """
    story.append(Paragraph(q4_ans, viva_a_style))

    story.append(PageBreak())

    # Q5: Dataset Engineering
    story.append(Paragraph("Q5: \"Tell us about your dataset engineering. Did you only test one recording or the whole dataset?\"", viva_q_style))
    q5_ans = """
    <b>Model Answer to Speak:</b><br/>
    \"We used the complete gold-standard <b>PhysioNet CHB-MIT Scalp EEG Database</b>, which consists of 24 pediatric subjects collected at Children's Hospital Boston.<br/>
    Rather than using a single file, we engineered a universal dataset manager (<code>src/eeg/chb_registry.py</code>) and a multi-threaded AWS S3 Open Data streaming engine (<code>scripts/download_s3.py</code>). We downloaded all <b>857 files totaling 42.62 GB</b> across all 24 subjects, including all 686 EDF recordings and all 141 recordings with documented seizures.<br/>
    Our multi-subject training engine (<code>scripts/train_eeg.py</code>) processes continuous recordings into 4-second windows (1,024 samples) with balanced BCE class weighting. On our multi-subject validation set of 5,875 windows, the 1D-CNN + BiLSTM model achieved <b>100.0% sensitivity</b> (111/111 seizure windows detected) and <b>99.8% specificity</b> (5,754/5,764 background windows correctly rejected).\"
    """
    story.append(Paragraph(q5_ans, viva_a_style))

    # =========================================================================
    # SECTION 5: MATHEMATICAL FORMULATIONS & EQUATIONS CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("5. Mathematical Formulations & Equations Cheat Sheet", h1_style))
    story.append(Paragraph(
        "Be prepared to write or explain these exact equations on the whiteboard during the viva:",
        body_style,
    ))

    math_data = [
        [Paragraph("<b>Concept / Task</b>", body_style), Paragraph("<b>Mathematical Formulation</b>", body_style), Paragraph("<b>Clinical Explanation</b>", body_style)],
        [
            Paragraph("<b>Class-Weighted BCE Loss (Signal Training)</b>", body_style),
            Paragraph("$$\\mathcal{L} = -\\left[ w_{pos} \\cdot y \\log \\hat{y} + (1-y) \\log(1-\\hat{y}) \\right]$$<br/>where $w_{pos} = \\frac{N_{neg}}{N_{pos}}$", body_style),
            Paragraph("Severe class imbalance: seizure windows account for <2% of total recording time. Weighting $w_{pos}$ penalizes missed seizures heavily, achieving 100% sensitivity.", body_style)
        ],
        [
            Paragraph("<b>Dense Vector Similarity (ChromaDB)</b>", body_style),
            Paragraph("$$\\text{sim}(q, d) = \\frac{\\mathbf{e}_q \\cdot \\mathbf{e}_d}{\\|\\mathbf{e}_q\\|_2 \\|\\mathbf{e}_d\\|_2}$$", body_style),
            Paragraph("Cosine similarity between query embedding $\\mathbf{e}_q$ and guideline passage $\\mathbf{e}_d$ mapped by BAAI/bge-small-en-v1.5 in $\\mathbb{R}^{384}$.", body_style)
        ],
        [
            Paragraph("<b>Query-Adaptive Activation Rule</b>", body_style),
            Paragraph("$$\\text{Activate}(m) = \\begin{cases} 1 & \\text{if } S_m(q) \\ge \\tau \\\\ 0 & \\text{if } S_m(q) < \\tau \\end{cases}$$", body_style),
            Paragraph("Threshold $\\tau = 0.50$. If the query does not ask for signal interpretation, $S_{eeg} < 0.5$, bypassing GPU neural feature extraction.", body_style)
        ],
        [
            Paragraph("<b>Self-RAG Overall Critique Score</b>", body_style),
            Paragraph("$$\\text{Critique}(y) = P([IsRel]) \\cdot P([IsSup]) \\cdot P([IsUse])$$", body_style),
            Paragraph("Joint probability of reflection tokens. If score $< 0.70$, refinement loop dynamically reduces $\\tau$ by 0.10 and re-retrieves.", body_style)
        ],
    ]
    t_math = Table(math_data, colWidths=[120, 200, 180])
    t_math.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_math)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 6: PRACTICAL DEMO & CONCLUSION
    # =========================================================================
    story.append(Paragraph("6. Live Demonstration Walkthrough & Examiner Wrap-Up", h1_style))
    story.append(Paragraph(
        "When examiners ask you to demonstrate the system on screen, guide them through these exact steps:",
        body_style,
    ))
    story.append(Paragraph("1. <b>Open the Live Web App:</b> Navigate to <code>https://prarit55.github.io/eeg/</code>. Point out the responsive dark-mode clinical UI.", bullet_style))
    story.append(Paragraph("2. <b>Show the EEG Upload Feature:</b> Drag and drop an EDF file or click <i>Case 1 (11y Female, Temporal Lobe Seizure)</i>. Show how the file badge immediately displays 23 channels, sampling rate (256 Hz), and recording duration.", bullet_style))
    story.append(Paragraph("3. <b>Demonstrate the 23-Channel Canvas:</b> Highlight the continuous multi-channel tracing and point out the red shaded ictal discharge window (2994s–3038s, 44 seconds duration). Explain that the BiLSTM detected the spike-and-wave burst with 100% confidence.", bullet_style))
    story.append(Paragraph("4. <b>Highlight Evidence-Grounded Citations:</b> Show the synthesized consultation report. Emphasize that every recommendation has an explicit citation handle ([Doc-1], [Doc-2], [KG-Triplets]), ensuring clinical transparency.", bullet_style))
    story.append(Paragraph("5. <b>Conclude Confidently:</b> <i>\"Respected panel, this project bridges the critical gap between raw biomedical time-series engineering and transparent clinical decision support, delivering factual, evidence-grounded AI for epilepsy care.\"</i>", bullet_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Mirror copy to project root
    import shutil
    shutil.copyfile(PDF_DEST, PDF_ROOT)
    print(f"[OK] Handbook PDF generated at:\n  - {PDF_DEST}\n  - {PDF_ROOT}")


if __name__ == "__main__":
    build_handbook()
