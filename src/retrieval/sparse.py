import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


BM25_PATH = Path("data/bm25_index.pkl")
CHUNKS_PATH = Path("data/chunks.json")


# keyword search using BM25
class SparseRetriever:

    # Load the BM25 index and chunk list from disk 
    def __init__(self):
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        # use chunks list to look up text and metadata by index position
        with open(CHUNKS_PATH, "r") as f:
            self.chunks = json.load(f)

    # Tokenize query and score chunks by keyword match
    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        # Lowercase and split 
        tokens = query_text.lower().split()

        # Get BM25 score for every chunk in the index
        scores = self.bm25.get_scores(tokens)

        # Sort chunk indexes by score (highest first) and take the top K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            # Skip chunks with zero score 
            if scores[i] > 0:
                results.append({
                    "id": self.chunks[i]["id"],
                    "text": self.chunks[i]["text"],
                    "metadata": self.chunks[i]["metadata"],
                    "score": float(scores[i]),
                })
        return results
