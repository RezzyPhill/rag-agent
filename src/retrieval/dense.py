from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = Path("data/chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:

    # Load the embedding model and connect to ChromaDB 
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(name="documents")

    # Convert the query to a vector, then ask ChromaDB for the most similar chunks
    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        # Embed the query into a vector using the same model used during ingestion
        query_vector = self.embedder.encode(query_text).tolist()

        # Ask ChromaDB for the top_k closest chunks by vector similarity
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Reformat ChromaDB's response into a list of dicts
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                # ChromaDB returns distance (lower = more similar), in score form(higher = better)
                "score": 1 - results["distances"][0][i],
            })
        return chunks
