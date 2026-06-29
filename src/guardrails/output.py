MIN_CONFIDENCE = 0.0  # floor below which we flag the answer as unreliable


# Validates the orchestrator's result dict before it is returned to the API caller.
# Returns (is_valid: bool, reason: str).
# Unlike the input guardrail this never blocks — is_valid=False adds a warning field
# to the response rather than rejecting it, because the user still deserves an answer.
def validate_output(result: dict) -> tuple[bool, str]:
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    confidence = result.get("confidence", 0.0)

    if not answer or not answer.strip():
        return False, "Answer is empty."

    if not citations:
        return False, "Answer contains no citations — cannot verify sources."

    if confidence <= MIN_CONFIDENCE:
        return False, f"Confidence too low to return a reliable answer ({confidence})."

    return True, ""
