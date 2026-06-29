from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


# One entry per chunk the answer drew from
class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page_num: int


class QuestionResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    iterations_used: int
    trace_id: str
    # for the output guardrail when the answer is returned but flagged
    warning: str | None = None
