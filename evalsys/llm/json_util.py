from __future__ import annotations

import json
import re
from typing import Any


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty LLM response")
    stripped = text.strip()
    candidates = [stripped]
    fenced = _FENCE.search(stripped)
    if fenced:
        candidates.insert(0, fenced.group(1))
    obj = _OBJECT.search(stripped)
    if obj:
        candidates.append(obj.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not parse JSON from LLM output: {last_error}")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_verdict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings") or []
    if isinstance(findings, str):
        findings = [findings]
    findings = [str(item).strip() for item in findings if str(item).strip()]
    grade = str(payload.get("grade") or "").strip().upper()[:1]
    if grade not in {"A", "B", "C", "D"}:
        grade = ""
    return {
        "score": clamp(float(payload.get("score", 0)), 0, 100),
        "confidence": clamp(float(payload.get("confidence", 0)), 0, 1),
        "grade": grade,
        "summary": str(payload.get("summary") or "").strip(),
        "findings": findings,
    }
