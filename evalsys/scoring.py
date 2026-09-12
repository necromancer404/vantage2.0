from __future__ import annotations

from evalsys.config import WeightSettings
from evalsys.schemas import AGENT_NAMES, AgentVerdict


def _letter_from_score(score: float, bands: dict[str, float]) -> str:
    ordered = sorted(bands.items(), key=lambda item: item[1], reverse=True)
    for letter, threshold in ordered:
        if score >= threshold:
            return letter
    return "D"


def combine_verdicts(
    verdicts: dict[str, AgentVerdict],
    weights: WeightSettings,
) -> tuple[float, float, str]:
    total = 0.0
    confidence_acc = 0.0
    scores: list[float] = []
    for name in AGENT_NAMES:
        verdict = verdicts[name]
        weight = weights.agents[name]
        total += weight * verdict.score
        confidence_acc += weight * verdict.confidence
        scores.append(verdict.score)

    spread = (max(scores) - min(scores)) / weights.max_score if scores else 0.0
    confidence = max(0.0, confidence_acc * (1.0 - weights.disagreement_spread_penalty * spread))
    total = min(weights.max_score, max(weights.min_score, total))
    grade = _letter_from_score(total, weights.grade_bands)
    return total, confidence, grade
