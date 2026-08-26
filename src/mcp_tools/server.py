from mcp.server.mcpserver import MCPServer

from src.retrieval.hybrid import HybridRetriever


mcp = MCPServer("rag-retrieval")


retriever = HybridRetriever()


@mcp.tool()
def search_documents(query: str) -> list[dict]:
    """Search the ingested document collection using hybrid (dense + BM25) retrieval.

    Returns the top matching chunks, each with its text, source document ID, and page number.
    """
    chunks = retriever.query(query)
    return [
        {
            "chunk_id": chunk["id"],
            "text": chunk["text"],
            "doc_id": chunk["metadata"].get("doc_id"),
            "page_num": chunk["metadata"].get("page_num"),
        }
        for chunk in chunks
    ]


if __name__ == "__main__":
    mcp.run()
