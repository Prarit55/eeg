"""
Configuration loader and settings schemas for the project.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

CONFIG_FILE_PATH = Path(__file__).resolve().parent / "default_config.yaml"


class ProjectConfig(BaseModel):
    name: str = "Query-Adaptive Multimodal Self-RAG for Epilepsy Analysis"
    version: str = "0.1.0"
    random_seed: int = 42


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_eeg_dir: str = "data/raw/eeg"
    raw_doc_dir: str = "data/raw/documents"
    raw_kg_dir: str = "data/raw/kg"
    processed_eeg_dir: str = "data/processed/eeg_windows"
    processed_kg_dir: str = "data/processed/kg_graph"
    vector_db_dir: str = "data/vector_db"
    mock_triplets_path: str = "data/raw/kg/optimus_kg_epilepsy.csv"
    sample_pdf_path: str = "data/raw/documents/sample_epilepsy.pdf"
    default_edf_path: str = "data/raw/eeg/chb01/chb01_03.edf"


class EEGConfig(BaseModel):
    sfreq: int = 256
    n_channels: int = 23
    window_sec: int = 4
    stride_sec: int = 2
    embed_dim: int = 128
    bandpass_low: float = 0.5
    bandpass_high: float = 60.0
    notch_freq: float = 60.0


class KnowledgeGraphConfig(BaseModel):
    triplets_path: str = "data/raw/kg/optimus_kg_epilepsy.csv"
    keywords: List[str] = Field(default_factory=lambda: ["epilepsy", "seizure", "temporal", "frontal"])
    default_hops: int = 1
    max_subgraph_nodes: int = 20


class VectorStoreConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3


class RoutingConfig(BaseModel):
    tau: float = 0.5
    extractor_backend: str = "auto"
    modules: Dict[str, str] = Field(
        default_factory=lambda: {"s_vector": "Vector", "s_graph": "Graph", "s_eeg": "EEG"}
    )


class SelfRAGConfig(BaseModel):
    max_refinement_steps: int = 2
    confidence_threshold: float = 0.6
    enable_critique: bool = True


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppSettings(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    eeg: EEGConfig = Field(default_factory=EEGConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    self_rag: SelfRAGConfig = Field(default_factory=SelfRAGConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: Optional[Path] = None) -> AppSettings:
    """Load configuration from YAML file or return defaults."""
    target_path = config_path or CONFIG_FILE_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return AppSettings(**raw_data)
    return AppSettings()


# Default singleton instance
config = load_config()
