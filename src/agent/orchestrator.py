import os
import uuid

from dotenv import load_dotenv

from src.agent.planner import Planner
from src.agent.reasoner import Reasoner
from src.agent.answer import Answerer
from src.retrieval.hybrid import HybridRetriever


load_dotenv()

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 3))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))


# Plan -> Retrieve -> Reason -> Refine -> Answer
class Orchestrator:

   
    def __init__(self):
        self.planner = Planner()
        self.retriever = HybridRetriever()
        self.reasoner = Reasoner()
        self.answerer = Answerer()

    # Run pipeline for a user question
    # Returns the final answer, citations, confidence level, iterations used, and trace ID
    def run(self, question: str) -> dict:
        # Generate a unique ID for this request so we can trace it through the logs
        trace_id = str(uuid.uuid4())

        print(f"[{trace_id}] Starting pipeline for: {question}")

        # Plan
        # break the question into focused sub-queries
        sub_queries = self.planner.plan(question)
        print(f"[{trace_id}] Plan produced {len(sub_queries)} sub-queries")

        # Retrive -> Reason -> Refine loop
        all_chunks = []       # accumulates retrieved chunks across all iterations
        confidence = 0.0
        iteration = 0

        while iteration < MAX_ITERATIONS:
            print(f"[{trace_id}] Iteration {iteration + 1}: retrieving...")

            # Retrieve chunks for each sub-query and add to the running collection
            for sub_query in sub_queries:
                chunks = self.retriever.query(sub_query["query"])
                all_chunks.extend(chunks)

            # Deduplicate cause the same chunk might be returned by multiple sub-queries
            all_chunks = _deduplicate(all_chunks)

            # Reason
            # given whats found, how confident? What's missing?
            confidence, missing, refined_queries = self.reasoner.evaluate(
                question=question,
                chunks=all_chunks,
            )
            print(f"[{trace_id}] Confidence after iteration {iteration + 1}: {confidence}")

            # Stop if confident enough
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"[{trace_id}] Confidence threshold met, stopping retrieval")
                break

            # Refine
            # Not confident yet, use the refined queries claude suggests and loop again
            if refined_queries:
                sub_queries = refined_queries
                print(f"[{trace_id}] Refining with {len(sub_queries)} new queries")

            iteration += 1

        # Answer
        print(f"[{trace_id}] Generating final answer...")
        result = self.answerer.answer(question=question, chunks=all_chunks)

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": confidence,
            "iterations_used": iteration + 1,
            "trace_id": trace_id,
        }


# keep the first occurrence of each chunk
def _deduplicate(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            unique.append(chunk)
    return unique
