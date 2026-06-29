import json
import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

ANSWER_SYSTEM_PROMPT = """You are a document question-answering assistant.

You will be given a question and a set of retrieved document chunks.

Your job is to produce a final, grounded answer using ONLY the information in the provided chunks.
Do not use any outside knowledge. If the chunks do not contain enough information to answer fully,
say so explicitly.

For every claim in your answer, cite the chunk(s) that support it using the chunk_id provided.

You must respond with valid JSON only, in this exact format:
{
  "answer": "Your complete answer here, written in clear prose.",
  "citations": [
    {"chunk_id": "...", "doc_id": "...", "page_num": 3}
  ]
}"""


# Generate the final grounded answer from the accumulated chunks
# Also runs the output grounding guardrail before returning
class Answerer:

    # Anthropic client
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = ANTHROPIC_MODEL

    # call claude to give cited answer, then verifies citations are grounded
    # Returns {"answer": str, "citations": list, "grounded": bool}
    def answer(self, question: str, chunks: list[dict]) -> dict:
        prompt = self._build_prompt(question, chunks)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw = response.content[0].text
        result = self._parse_response(raw)

        # Run the grounding check and attach the result
        result["grounded"] = self._grounding_check(result["citations"], chunks)

        return result

    # Formats chunks with their IDs and source metadata so claude can cite them precisely
    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        chunk_lines = []
        for chunk in chunks:
            doc_id = chunk["metadata"].get("doc_id", "unknown")
            page = chunk["metadata"].get("page_num", "?")
            chunk_lines.append(
                f"[chunk_id: {chunk['id']} | source: {doc_id}, page {page}]\n{chunk['text']}"
            )

        chunks_block = "\n\n".join(chunk_lines)

        return (
            f"Question:\n{question}\n\n"
            f"Retrieved chunks:\n{chunks_block}\n\n"
            "Answer the question using only the chunks above. Cite every claim."
        )

    # Parse claude's JSON response 
    # Falls back to "insufficient information" answer 
    def _parse_response(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
            return {
                "answer": str(data["answer"]),
                "citations": data.get("citations", []),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "answer": "Unable to generate an answer due to an internal error.",
                "citations": [],
            }

    # verifies every cited chunk_id was actually retrieved
    def _grounding_check(self, citations: list[dict], chunks: list[dict]) -> bool:
        valid_ids = {chunk["id"] for chunk in chunks}
        ungrounded = [c for c in citations if c.get("chunk_id") not in valid_ids]

        if ungrounded:
            # Log
            print(f"[grounding_check] WARNING: {len(ungrounded)} citation(s) reference unknown chunk IDs: "
                  f"{[c.get('chunk_id') for c in ungrounded]}")
            return False

        return True
