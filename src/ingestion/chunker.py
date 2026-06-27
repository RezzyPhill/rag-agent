import re
from dataclasses import dataclass, field

# Expanding acronyms before embedding ensures semantic search finds chunks
# that use the abbreviated form when a user queries the full term.
ACRONYM_MAP = {
    "DoD": "Department of Defense",
    "MBSE": "Model-Based Systems Engineering",
    "SE": "Systems Engineering",
    "ME": "Mission Engineering",
    "SoS": "System of Systems",
    "CONOPS": "Concept of Operations",
    "MOE": "Measure of Effectiveness",
    "MOP": "Measure of Performance",
    "IOC": "Initial Operating Capability",
    "FOC": "Full Operating Capability",
    "JCIDS": "Joint Capabilities Integration and Development System",
    "ICD": "Initial Capabilities Document",
    "CDD": "Capability Development Document",
    "CPD": "Capability Production Document",
    "DODAF": "Department of Defense Architecture Framework",
    "COI": "Community of Interest",
    "OPSEC": "Operational Security",
    "ISR": "Intelligence Surveillance and Reconnaissance",
    "C2": "Command and Control",
    "KPP": "Key Performance Parameter",
    "CDR": "Critical Design Review",
    "PDR": "Preliminary Design Review",
}


@dataclass
class Chunk:
    text: str
    doc_id: str
    page_num: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def expand_acronyms(text: str) -> str:
    for acronym, expansion in ACRONYM_MAP.items():
        text = re.sub(rf"\b{re.escape(acronym)}\b", expansion, text)
    return text


def chunk_text(
    text: str,
    doc_id: str,
    page_num: int,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    text = expand_acronyms(text)
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]

        if piece.strip():
            chunks.append(
                Chunk(
                    text=piece,
                    doc_id=doc_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    metadata={
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

        start = end - overlap

    return chunks
