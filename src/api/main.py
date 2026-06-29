from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.agent.orchestrator import Orchestrator
from src.api.models import Citation, QuestionRequest, QuestionResponse
from src.guardrails.input import InputGuardrail
from src.guardrails.output import validate_output
from src.observability.tracing import configure_logging, get_logger


# Runs once at startup
# Load models and indexes so first real request isn't slow
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.orchestrator = Orchestrator()
    app.state.input_guard = InputGuardrail()
    get_logger().info("startup_complete")
    yield


app = FastAPI(
    title="RAG Agent",
    description="Agentic RAG system: plan → retrieve → reason → refine → answer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    log = get_logger()

    # Input guardrail 
    is_valid, reason = app.state.input_guard.validate(request.question)
    if not is_valid:
        log.warning("input_rejected", reason=reason)
        raise HTTPException(status_code=400, detail=reason)

    # Run the full pipeline
    result = app.state.orchestrator.run(question=request.question)

    # Output guardrail 
    output_ok, warning = validate_output(result)

    # citation dicts from claude into typed citation models
    citations = [
        Citation(
            chunk_id=c.get("chunk_id", ""),
            doc_id=c.get("doc_id", ""),
            page_num=int(c.get("page_num", 0)),
        )
        for c in result.get("citations", [])
    ]

    return QuestionResponse(
        answer=result["answer"],
        citations=citations,
        confidence=result["confidence"],
        iterations_used=result["iterations_used"],
        trace_id=result["trace_id"],
        warning=warning if not output_ok else None,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
