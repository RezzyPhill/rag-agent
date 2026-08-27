import asyncio
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.guardrails.input import InputGuardrail


load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Project root -- so the server subprocess can resolve `python -m src.mcp_tools.server`
# regardless of where this script itself is launched from.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# Spawn the MCP server as a local subprocess, using the same Python interpreter
# (same venv) this client is running under -- stdio transport, no network involved.
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.mcp_tools.server"],
    cwd=PROJECT_ROOT,
)

AGENT_SYSTEM_PROMPT = """You are a document question-answering assistant with access to a
search_documents tool over an internal knowledge base. Use the tool as many times as needed
to gather enough information, then answer the question using only what the tool returns.
Cite the chunk_id for every claim."""


# Runs one question through a real Claude tool-calling loop, where the tool
# execution step goes through an MCP server instead of a direct Python call.
# Returns a plain string in all cases -- either the answer, or a clear
# explanation of what went wrong, never a raw traceback.
async def run(question: str) -> str:
    # Fail-closed, same reasoning as the original pipeline's InputGuardrail:
    # reject before spending a subprocess spawn + tool-calling loop on bad input.
    guardrail = InputGuardrail()
    is_valid, reason = guardrail.validate(question)
    if not is_valid:
        return f"Rejected by input guardrail: {reason}"

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Ask the MCP server what tools it offers, convert to Claude's tool schema.
                # The shapes already line up: name/description/input_schema on both sides.
                discovered = await session.list_tools()
                claude_tools = [
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in discovered.tools
                ]

                messages = [{"role": "user", "content": question}]

                while True:
                    response = client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=1024,
                        system=AGENT_SYSTEM_PROMPT,
                        tools=claude_tools,
                        messages=messages,
                    )

                    messages.append({"role": "assistant", "content": response.content})

                    # Claude stops asking for tools -> it's ready to answer, return the text.
                    if response.stop_reason != "tool_use":
                        return "".join(
                            block.text for block in response.content if block.type == "text"
                        )

                    # Claude asked for one or more tool calls -- run each one against
                    # the MCP server (not a local function call) and report results back.
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        result = await session.call_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result.structured_content),
                        })

                    messages.append({"role": "user", "content": tool_results})
    except Exception as e:
        # Covers: the server subprocess failing to start, dying mid-run, or the
        # stdio pipe breaking. Surface a clear message instead of a raw traceback --
        # this is a demo script, not a service, so "try again" is a reasonable answer.
        return f"MCP session failed ({type(e).__name__}: {e}). The server subprocess may have crashed -- try again."


if __name__ == "__main__":
    # Windows' console defaults to cp1252, which can't display every character
    # Claude might produce (emoji, some punctuation) -- force UTF-8 for stdout.
    sys.stdout.reconfigure(encoding="utf-8")

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this knowledge base about?"
    print(asyncio.run(run(question)))
