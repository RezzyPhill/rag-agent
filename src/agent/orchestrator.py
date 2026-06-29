import os
import uuid

from dotenv import load_dotenv

from src.agent.planner import Planner
from src.agent.reasoner import Reasoner
from src.agent.answer import Answerer
from src.observability.tracing import get_logger
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
        # Generate a unique ID for this request so every log line can be tied together
        trace_id = str(uuid.uuid4())
        log = get_logger(trace_id=trace_id)

        log.info("pipeline_start", question=question)

        # Plan
        # break the question into focused sub-queries
        sub_queries = self.planner.plan(question)
        log.info("plan_complete", sub_query_count=len(sub_queries))

        # Retrieve -> Reason -> Refine loop
        all_chunks = []       # accumulates retrieved chunks across all iterations
        confidence = 0.0
        iteration = 0

        while iteration < MAX_ITERATIONS:
            log.info("retrieval_start", iteration=iteration + 1)

            # Retrieve chunks for each sub-query and add to the running collection
            for sub_query in sub_queries:
                chunks = self.retriever.query(sub_query["query"])
                all_chunks.extend(chunks)

            # Deduplicate cause the same chunk might be returned by multiple sub-queries
            all_chunks = _deduplicate(all_chunks)

            # Reason
            # given what's found, how confident? What's missing?
            confidence, missing, refined_queries = self.reasoner.evaluate(
                question=question,
                chunks=all_chunks,
            )
            log.info("reason_complete", iteration=iteration + 1, confidence=confidence, missing=missing)

            # Stop if confident enough
            if confidence >= CONFIDENCE_THRESHOLD:
                log.info("confidence_threshold_met", confidence=confidence)
                break

            # Refine 
            # Not confident yet, use claude's refined queries and loop again
            if refined_queries:
                sub_queries = refined_queries
                log.info("refine", new_query_count=len(sub_queries))

            iteration += 1

        # Answer 
        log.info("answer_start")
        result = self.answerer.answer(question=question, chunks=all_chunks)
        log.info("pipeline_complete", grounded=result.get("grounded"), iterations_used=iteration + 1)

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": confidence,
            "iterations_used": iteration + 1,
            "trace_id": trace_id,
        }


# Keep the first occurrence of each chunk 
def _deduplicate(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            unique.append(chunk)
    return unique
