# Query-Adaptive Multimodal Self-RAG for Evidence-Grounded Medical Reasoning in Epilepsy Analysis

> **Minor Project [ARP-455]**  
> **Author**: Parth Saini (06219051923)  
> **Supervisor**: Dr. Ruchika Lalit, Assistant Professor, USAR  

---

## 1. Project Overview

This project implements a **query-adaptive multimodal Self-RAG framework** for evidence-grounded reasoning over EEG signals and epilepsy-related medical knowledge. Conventional medical AI systems focus solely on signal-level seizure detection or general text-based question answering without patient signal awareness. Furthermore, standard RAG frameworks inefficiently query all retrieval sources for every question regardless of the actual information need.

This system introduces:
1. **Multimodal Patient Context & EEG Processing**: Neural encoding (1D-CNN + BiLSTM) of continuous scalp EEG recordings (CHB-MIT database).
2. **Complementary Knowledge Channels**:
   - **Vector Store**: Medical literature, ILAE guidelines, and clinical criteria.
   - **Knowledge Graph**: Epilepsy/neurology ontology representing structured relations between brain lobes, seizure types, symptoms, and antiseizure medications (ASMs).
3. **Query Understanding & Adaptive Routing**: Intent classification and constrained decomposition into atomic subqueries, with dynamic source suitability scoring ($S_{vector}, S_{graph}, S_{eeg}$) activating only channels meeting a threshold $\tau$.
4. **Multimodal Evidence Fusion**: Aggregation of patient history, EEG features, graph subgraphs, and text passages into a structured reasoning context.
5. **Self-RAG Reflection & Grounding**: Self-critique checking relevance ($[IsRel]$), answer support ($[IsSup]$), and utility ($[IsUse]$) with citation grounding and confidence scoring.

---

## 2. File System & Architecture

The repository is organized into a clean, modular structure mirroring the architecture defined in the project synopsis:

```
d:/epilepsy/
├── configs/                              # Central configurations & hyperparameters
│   ├── default_config.yaml              # Thresholds (tau=0.5), paths, model specs, dimensions
│   └── settings.py                      # Pydantic Settings schema & YAML loader
│
├── data/                                 # Multi-tier data hierarchy
│   ├── raw/
│   │   ├── eeg/                         # Scalp EEG recordings (.edf, .seizures, summaries)
│   │   ├── documents/                   # Medical PDFs, ILAE guidelines, StatPearls
│   │   └── kg/                          # Triplet CSVs (PrimeKG / mock triplets)
│   ├── processed/
│   │   ├── eeg_windows/                 # Cached segmented windows & embeddings
│   │   └── kg_graph/                    # Cached NetworkX graph indices
│   └── vector_db/                       # ChromaDB persistent vector storage
│
├── src/                                  # Core Application Package
│   ├── eeg/                             # Module 1: Scalp EEG Processing & Neural Encoding
│   │   ├── loader.py                    # EDF reading, MNE bandpass (0.5-60Hz) & notch filtering
│   │   ├── preprocessor.py              # 4s windowing (50% stride), z-score normalization
│   │   ├── models.py                    # 1D-CNN + BiLSTM encoder (logits & 128-d embeddings)
│   │   └── pipeline.py                  # EEGPipeline runner & synthetic EEG generator
│   │
│   ├── knowledge/                       # Modules 2 & 3: External Knowledge Sources
│   │   ├── knowledge_graph.py           # EpilepsyKnowledgeGraph (MultiDiGraph) & triplet loader
│   │   ├── graph_retriever.py           # Entity resolution, k-hop subgraphs & path discovery
│   │   ├── document_loader.py           # PDF loader & recursive character text splitter
│   │   └── vector_store.py              # ChromaDB vector store with HuggingFace embeddings
│   │
│   ├── routing/                         # Module 4: Query Understanding & Adaptive Routing
│   │   ├── schemas.py                   # Pydantic schemas (RoutingScores, SubQueryRoute, etc.)
│   │   ├── classifier.py                # Intent classifier (definitional, relational, eeg, complex)
│   │   ├── decomposer.py                # Constrained query decomposition into atomic subqueries
│   │   ├── scorer.py                    # Suitability scoring (S_vector, S_graph, S_eeg vs tau)
│   │   ├── patient_context.py           # HistoryExtractor (NER/LLM/lexicon) & de-identification
│   │   └── router.py                    # Unified QueryRouter integrating decomposition & scoring
│   │
│   ├── fusion/                          # Module 5: Multimodal Evidence Fusion
│   │   ├── aggregator.py                # EvidenceAggregator combining signals, triples, chunks
│   │   └── context_formatter.py         # Formats structured multimodal context with citation tags
│   │
│   ├── self_rag/                        # Module 6: Self-RAG Reflection & Medical Reasoning
│   │   ├── critique.py                  # Reflection critic ([IsRel], [IsSup], [IsUse])
│   │   ├── generator.py                 # Evidence-grounded generation with citation tracking
│   │   └── loop.py                      # Orchestrates routing, retrieval, fusion & critique
│   │
│   └── api/                             # Module 7: Backend Service
│       ├── models.py                    # Request & response Pydantic models
│       ├── routes.py                    # API endpoints (/api/v1/query, /api/v1/eeg/analyze, etc.)
│       └── app.py                       # FastAPI application factory
│
├── evaluation/                          # Benchmarking Suite (Objective 7)
│   ├── benchmarks/
│   │   └── test_cases.json              # Curated clinical test questions with ground truth routes
│   ├── eval_router.py                   # Routing accuracy & history invariance benchmark
│   └── eval_retrieval.py                # Adaptive vs. Non-Adaptive retrieval efficiency benchmark
│
├── tests/                               # Comprehensive Automated Test Suite
│   ├── test_eeg_pipeline.py             # Signal generation, windowing & neural forward pass tests
│   ├── test_knowledge_graph.py          # Entity search & k-hop subgraph traversal tests
│   ├── test_patient_context.py          # NER/LLM extraction, de-identification & fallback tests
│   ├── test_query_router.py             # Intent classification & decomposition tests
│   ├── test_fusion.py                   # Multimodal evidence aggregation & formatting tests
│   └── test_self_rag.py                 # Reflection critique & citation generation tests
│
├── scripts/                             # CLI Utilities
│   ├── run_pipeline.py                  # End-to-end reasoning CLI runner
│   ├── build_knowledge_graph.py         # Triplet parsing and graph metric inspector
│   └── build_vector_store.py            # PDF ingestion & Chroma indexing script
│
├── main.py                              # Root CLI entrypoint (backward-compatible facade)
├── eval_router.py                       # Root routing evaluation facade
├── test_patient_context.py              # Root patient context test facade
└── requirements.txt                     # Project dependencies
```

---

## 3. Quick Start & Execution

### 1. Run the Full Reasoning Pipeline (CLI)
Test end-to-end query decomposition, adaptive source activation, multimodal evidence fusion, and Self-RAG reflection:
```powershell
python scripts/run_pipeline.py
```

### 2. Run the Retrieval Efficiency Benchmark
Directly measures the reduction in redundant module calls achieved by query-adaptive routing compared to non-adaptive RAG:
```powershell
python evaluation/eval_retrieval.py
```

### 3. Run Routing & Invariance Tests
```powershell
python eval_router.py
```

### 4. Run the Full Unit Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Where to Code Next (Gantt Chart Milestones)

| Milestone / Layer | Target Files to Extend | Planned Features |
| :--- | :--- | :--- |
| **EEG Weights & Feature Bank** | `src/eeg/pipeline.py`, `src/eeg/models.py` | Train `EEGEncoder` on full CHB-MIT recordings and save model weights to `data/processed/`. |
| **Expanded Knowledge Graph** | `src/knowledge/knowledge_graph.py` | Ingest PrimeKG full CSV triplets or connect Neo4j backend as outlined in the synopsis. |
| **Full LLM Generation** | `src/self_rag/generator.py` | Connect local Open-Source LLMs (via Hugging Face or Ollama) or OpenAI/Gemini models for full medical synthesis. |
| **FastAPI Service Deployment** | `src/api/app.py`, `src/api/routes.py` | Launch `uvicorn src.api.app:create_app --factory --reload` for interactive frontend/API integration. |
