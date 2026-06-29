import json
import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


PLANNER_SYSTEM_PROMPT = """You are a query planning assistant for a document retrieval system.

Your job is to decompose a user's question into clear sub-questions that can be used to search a knowledge base.

Rules:
- Break the question into 1-5 focused sub-questions
- Tag each as "parallel" (can be retrieved at the same time) or "sequential" (depends on a previous answer)
- For sequential sub-questions, specify which sub-question ID they depend on
- Keep sub-questions specific and searchable

You must respond with valid JSON only, in this exact format:
{
  "sub_queries": [
    {"id": 1, "query": "...", "type": "parallel", "depends_on": null},
    {"id": 2, "query": "...", "type": "sequential", "depends_on": 1}
  ]
}"""


# breaks a user question into a structured retrieval plan that returns a list of sub-queries with dependency tags
class Planner:

    # Anthropic client 
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = ANTHROPIC_MODEL

 
    def plan(self, question: str) -> list[dict]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Decompose this question into sub-queries:\n\n{question}"}
            ],
        )

        # Extract raw text from claude response
        raw = response.content[0].text

        # claude JSON strip 
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        # Parse JSON, fall back to treating the whole question as a single parallel sub-query
        try:
            plan = json.loads(raw)
            return plan["sub_queries"]
        except (json.JSONDecodeError, KeyError):
            return [{"id": 1, "query": question, "type": "parallel", "depends_on": None}]
