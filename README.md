# RAG Agent

An agentic RAG system that answers complex questions over an internal knowledge base by planning, iteratively retrieving, and reasoning — rather than relying on single-shot retrieval.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  INPUT GUARDRAIL                │
│  Length check + relevance check │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  PHASE 1: PLAN                  │
│  Decompose into sub-queries     │
│  tagged [parallel]/[sequential] │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐   ◄──────────────────────┐
│  PHASE 2: RETRIEVE              │                          │
│  Hybrid search per sub-query    │                          │
│  Dense (ChromaDB) + BM25 → RRF  │                          │
└─────────────────────────────────┘                          │
     │                                                        │
     ▼                                                        │
┌─────────────────────────────────┐                          │
│  PHASE 3: REASON                │                          │
│  Confidence score (0–1)         │                          │
│  + identify gaps                │                          │
└─────────────────────────────────┘                          │
     │                                                        │
     ├── confidence ≥ 0.7 OR max_iterations ──────────┐      │
     │                                                 │      │
     ▼                                                 │      │
┌─────────────────────────────────┐                   │      │
│  PHASE 4: REFINE                │───────────────────┘      │
│  Generate improved queries      │  loop back to RETRIEVE   │
└─────────────────────────────────┘                          │
                                                             │
     ┌───────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  PHASE 5: ANSWER                │
│  Grounded answer with citations │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  OUTPUT GUARDRAIL               │
│  Verify citations + confidence  │
└─────────────────────────────────┘
     │
     ▼
{ answer, citations, confidence, iterations_used, trace_id }
```

## Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Claude (`claude-sonnet-4-6`) | Reasoning quality in planner and reasoner |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, local, no extra API key |
| Vector store | ChromaDB | In-process, no infrastructure to manage |
| Sparse index | `rank-bm25` | Catches exact keyword matches dense search misses |
| API | FastAPI | Async, automatic OpenAPI docs |
| Observability | `structlog` + UUID trace IDs | Structured JSON logs, every request traceable |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Running

**Step 1 — Ingest documents** (run once)

Place PDFs in `data/documents/`, then:

```bash
python -m src.ingestion.pipeline
```

This builds the ChromaDB vector index and BM25 keyword index from your documents.

**Step 2 — Start the server**

```bash
uvicorn src.api.main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## API

### `POST /ask`

```json
{
  "question": "What is mission engineering?"
}
```

Response:

```json
{
  "answer": "Mission engineering is...",
  "citations": [
    { "chunk_id": "idt1_p2_c3", "doc_id": "idt1", "page_num": 2 }
  ],
  "confidence": 0.85,
  "iterations_used": 2,
  "trace_id": "863b8894-...",
  "warning": null
}
```

`warning` is `null` on a clean response. If the output guardrail flags the answer (low confidence, missing citations), `warning` contains a human-readable explanation and the answer is still returned.

### `GET /health`

```json
{ "status": "ok" }
```

## Design Decisions

**Hybrid retrieval (Dense + BM25 + RRF)**
Defense and engineering documents are acronym-heavy with precise terminology. Dense-only retrieval misses exact keyword matches. BM25 catches them. Reciprocal Rank Fusion (RRF) merges the two result sets by rank position rather than score, because ChromaDB similarity scores and BM25 scores live on incompatible scales.

**Structured phases with adaptive stopping**
The agent enforces the high-level shape (plan → retrieve → reason → refine → answer) but adapts within it. The retrieval loop runs 1–3 times based on the reasoner's confidence score. A hard `MAX_ITERATIONS` cap acts as a deterministic safety net alongside the LLM's judgment.

**Acronym expansion at ingestion**
Common defense acronyms (OPSEC, DoD, SoS, etc.) are expanded before embedding. This ensures semantic search connects queries using full terms to chunks that only contain abbreviations.

**Guardrails at both boundaries**
Input guardrail fails closed (blocks irrelevant questions before any pipeline cost is incurred). Output guardrail fails open (flags low-quality answers but returns them, since the user already paid the latency).

**Custom orchestration over LangChain**
Every component is written directly against the Anthropic SDK. This means every design choice is explainable from first principles rather than being delegated to framework internals.
