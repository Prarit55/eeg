"""
Module 2 - Medical vector database (PDF -> chunks -> Chroma store).

Embeddings: BAAI/bge-small-en-v1.5 via HuggingFace (downloaded on first build()).
`create_sample_pdf` writes a small epilepsy PDF so the module runs out of the box;
point `pdf_paths` at StatPearls / ILAE PDFs for real use.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.knowledge.document_loader import SAMPLE_PDF_PATH, create_sample_pdf


DEFAULT_DB_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "vector_db")
DEFAULT_COLLECTION_NAME = "epilepsy_medical_guidelines"


class MedicalVectorStore:
    """PDF loader + splitter + Chroma store. Embeddings load lazily."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        device: str = "cpu",
        persist_directory: Optional[str] = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.embedding_model = embedding_model
        self.device = device
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.store: Optional[Chroma] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def get_store(self) -> Chroma:
        """Get or lazily connect to the Chroma vector store."""
        if self.store is not None:
            return self.store
        if self.persist_directory and (Path(self.persist_directory) / "chroma.sqlite3").exists():
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=self.persist_directory,
            )
            return self.store
        raise RuntimeError("Vector store is empty - call build() first.")

    # -- ingestion ---------------------------------------------------------- #
    def load_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Load every page of every PDF as a Document (metadata: source, page)."""
        docs: List[Document] = []
        for p in pdf_paths:
            docs.extend(PyPDFLoader(p).load())
        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

    def build(self, pdf_paths: Optional[List[str]] = None) -> int:
        """Load -> chunk -> embed -> index. Returns the number of chunks indexed."""
        if not pdf_paths:
            pdf_paths = [SAMPLE_PDF_PATH]
        for p in pdf_paths:
            if not Path(p).exists():
                if p == SAMPLE_PDF_PATH:
                    create_sample_pdf(p)
                else:
                    raise FileNotFoundError(p)

        chunks = self.split(self.load_pdfs(pdf_paths))
        embeddings = self._get_embeddings()
        
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        return len(chunks)

    # -- retrieval ---------------------------------------------------------- #
    def retrieve_text(self, query: str, k: int = 3) -> List[str]:
        """Top-k chunk texts for `query`."""
        store = self.get_store()
        return [d.page_content for d in store.similarity_search(query, k=k)]

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """Top-k langchain Document objects for `query`."""
        store = self.get_store()
        return store.similarity_search(query, k=k)
