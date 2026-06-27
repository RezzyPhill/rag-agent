import json
import pickle
from pathlib import Path

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import chunk_text

DOCS_DIR = Path("data/documents")
CHROMA_DIR = Path("data/chroma_db")
BM25_PATH = Path("data/bm25_index.pkl")
CHUNKS_PATH = Path("data/chunks.json")

EMBED_MODEL = "all-MiniLM-L6-v2"


def ingest():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name="documents")

    all_chunks = []

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        print(f"Processing {doc_id}...")
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            chunks = chunk_text(text, doc_id=doc_id, page_num=page_num)
            all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    print("Embedding and storing in ChromaDB...")
    texts = [c.text for c in all_chunks]
    ids = [f"{c.doc_id}_p{c.page_num}_c{c.chunk_index}" for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

    print("Building BM25 index...")
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    with open(CHUNKS_PATH, "w") as f:
        json.dump(
            [{"id": ids[i], "text": texts[i], "metadata": metadatas[i]} for i in range(len(all_chunks))],
            f,
            indent=2,
        )

    print("Ingestion complete.")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  BM25 index: {BM25_PATH}")
    print(f"  Chunks: {CHUNKS_PATH}")


if __name__ == "__main__":
    ingest()