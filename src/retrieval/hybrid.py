import os

from dotenv import load_dotenv

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


load_dotenv()

# results to fetch from retrivers before merging
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 5))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", 5))
# final results to return after merging
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))

# RRF constant 
RRF_K = 60


# Combines semantic, keyword search using RRF
class HybridRetriever:

    
    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    # Run both searches, then merge results using RRF scoring
    def query(self, query_text: str) -> list[dict]:
        dense_results = self.dense.query(query_text, top_k=TOP_K_DENSE)
        sparse_results = self.sparse.query(query_text, top_k=TOP_K_SPARSE)

        # Dictionary to accumulate RRF scores per chunk ID
        rrf_scores = {}

        # Score each dense result by its rank position
        # Higher rank = lower score
        for rank, chunk in enumerate(dense_results):
            cid = chunk["id"]
            rrf_scores.setdefault(cid, {"chunk": chunk, "score": 0.0})
            rrf_scores[cid]["score"] += 1 / (RRF_K + rank + 1)

        # Add sparse scores on top, chunks appearing in both lists accumulate more score
        for rank, chunk in enumerate(sparse_results):
            cid = chunk["id"]
            rrf_scores.setdefault(cid, {"chunk": chunk, "score": 0.0})
            rrf_scores[cid]["score"] += 1 / (RRF_K + rank + 1)

        # Sort all chunks by their total RRF score, highest first
        ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        # Return only the chunk data for the top K results
        return [entry["chunk"] for entry in ranked[:TOP_K_FINAL]]
