import json
import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 1000

RELEVANCE_SYSTEM_PROMPT = """You are a relevance filter for a document question-answering system.
The knowledge base contains internal technical and policy documents for a defense organization
(architecture notes, engineering standards, operational policies, product documentation).

Decide whether the user's question is on-topic for this knowledge base.

Respond with valid JSON only:
{"relevant": true, "reason": "..."}
or
{"relevant": false, "reason": "..."}"""


# Validate a question before it enters the pipeline
# API uses the reason as the error message 
class InputGuardrail:

    # Anthropic client
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = ANTHROPIC_MODEL

    # Entry point runs length check first then LLM relevance check 
    # Short circuits on length failure so we never pay for obviously bad input
    def validate(self, question: str) -> tuple[bool, str]:
        length_ok, length_reason = self._check_length(question)
        if not length_ok:
            return False, length_reason

        return self._check_relevance(question)

    # Deterministic check 
    def _check_length(self, question: str) -> tuple[bool, str]:
        if len(question) < MIN_QUESTION_LENGTH:
            return False, f"Question is too short (minimum {MIN_QUESTION_LENGTH} characters)."
        if len(question) > MAX_QUESTION_LENGTH:
            return False, f"Question is too long (maximum {MAX_QUESTION_LENGTH} characters)."
        return True, ""

    # LLM based relevance check  keeps max_tokens low since we only need a yes/no
    def _check_relevance(self, question: str) -> tuple[bool, str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            system=RELEVANCE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": question}
            ],
        )

        raw = response.content[0].text

        try:
            data = json.loads(raw)
            is_relevant = bool(data["relevant"])
            reason = str(data.get("reason", ""))
            if not is_relevant:
                return False, f"Question does not appear relevant to the knowledge base: {reason}"
            return True, ""
        except (json.JSONDecodeError, KeyError):
            # If the relevance check itself fails, let the request through rather than
            # blocking valid questions due to a guardrail malfunction
            return True, ""
