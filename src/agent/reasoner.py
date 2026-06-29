import json
import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

REASONER_SYSTEM_PROMPT = """You are a reasoning assistant for a document retrieval system.

You will be given a user question and a set of retrieved document chunks.

Your job is to:
1. Evaluate how well the retrieved chunks answer the question
2. Assign a confidence score from 0.0 to 1.0
3. Identify what information is still missing (if any)
4. Suggest refined search queries to fill the gaps (only if confidence is below 0.7)

Rules:
- Be honest about gaps — do not claim confidence you do not have
- refined_queries must be [] if confidence >= 0.7
- Each refined query uses the same format as the planner output

You must respond with valid JSON only, in this exact format:
{
  "confidence": 0.85,
  "missing": "No significant gaps found",
  "refined_queries": []
}"""


# Evaluates retrieved chunks against the original question
# Returns a confidence score, a description of what is missing, and refined queries
class Reasoner:

    # Anthropic client
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = ANTHROPIC_MODEL

    # Main entry point called by the orchestrator after each retrieval round
    # Returns (confidence: float, missing: str, refined_queries: list[dict])
    def evaluate(self, question: str, chunks: list[dict]) -> tuple[float, str, list[dict]]:
        prompt = self._build_prompt(question, chunks)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=REASONER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw = response.content[0].text
        return self._parse_response(raw)

    # Formats the question and retrieved chunks into a single prompt string
    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        chunk_lines = []
        for i, chunk in enumerate(chunks, start=1):
            # Include source metadata so claude can assess coverage across documents
            doc_id = chunk["metadata"].get("doc_id", "unknown")
            page = chunk["metadata"].get("page_num", "?")
            chunk_lines.append(
                f"[Chunk {i} | source: {doc_id}, page {page}]\n{chunk['text']}"
            )

        chunks_block = "\n\n".join(chunk_lines)

        return (
            f"Question:\n{question}\n\n"
            f"Retrieved chunks:\n{chunks_block}\n\n"
            "Evaluate whether these chunks are sufficient to answer the question."
        )

    # Parses Claude's JSON response into the three return values.
    # Falls back to confidence=0.0 + original-question pass through 
    def _parse_response(self, raw: str) -> tuple[float, str, list[dict]]:
        # claude JSON strip 
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        try:
            data = json.loads(raw)
            confidence = float(data["confidence"])
            missing = str(data.get("missing", ""))
            refined_queries = data.get("refined_queries", [])
            return confidence, missing, refined_queries
        except (json.JSONDecodeError, KeyError, ValueError):
            # Degraded state: treat as zero confidence so the loop continues
            return 0.0, "Failed to parse reasoner response", []
