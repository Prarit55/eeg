"""
Build and index medical guidelines in the Chroma vector store.
Scans `data/raw/documents/` for clinical PDFs, chunks text, computes embeddings,
and persists the index to `data/vector_db/`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.vector_store import MedicalVectorStore

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "documents"
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_db"


def main():
    parser = argparse.ArgumentParser(description="Medical Vector Store Builder")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Directory containing medical PDFs")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Directory to persist Chroma vector DB")
    parser.add_argument("--test-query", default="What are first-line drugs for focal seizures?", help="Sample query to test retrieval")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    pdf_files = sorted([str(p) for p in docs_dir.glob("*.pdf")])

    if not pdf_files:
        print(f"No PDF files found in {docs_dir}. Generating sample documents...")
        from src.knowledge.document_loader import create_sample_pdf
        pdf_files = [create_sample_pdf(str(docs_dir / "sample_epilepsy.pdf"))]

    print("=" * 80)
    print("MEDICAL GUIDELINE VECTOR STORE INGESTION")
    print(f"Document sources ({len(pdf_files)} files):")
    for p in pdf_files:
        print(f"  - {Path(p).name} ({Path(p).stat().st_size / 1024:.1f} KB)")
    print(f"Chroma persistence directory: {args.db_dir}")
    print("=" * 80)

    vs = MedicalVectorStore(persist_directory=args.db_dir)
    print("\nExtracting, chunking, and embedding documents (BAAI/bge-small-en-v1.5)...")
    n_chunks = vs.build(pdf_files)
    print(f"Successfully indexed {n_chunks} chunks into persistent Chroma vector store.")

    # Verification retrieval
    print("\n--- Testing Retrieval Grounding ---")
    print(f"Query: '{args.test_query}'")
    hits = vs.retrieve_text(args.test_query, k=2)
    for i, h in enumerate(hits, 1):
        clean_text = h.strip().replace("\n", " ")
        print(f"\n[Hit {i}]: {clean_text[:180]}...")
    print("\n" + "=" * 80)
    print("Vector Store build and verification completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
