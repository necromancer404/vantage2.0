from __future__ import annotations

import math

from evalsys.config import WeightSettings
from evalsys.schemas import AGENT_NAMES, AgentVerdict


def round_marks(value: float, ceiling: float) -> int:
    if value >= 0:
        rounded = int(math.floor(value + 0.5))
    else:
        rounded = int(math.ceil(value - 0.5))
    high = int(math.floor(ceiling + 0.5)) if ceiling >= 0 else 0
    return max(0, min(high, rounded))


def _letter_from_score(score: float, bands: dict[str, float]) -> str:
    ordered = sorted(bands.items(), key=lambda item: item[1], reverse=True)
    for letter, threshold in ordered:
        if score >= threshold:
            return letter
    return "D"


def combine_verdicts(
    verdicts: dict[str, AgentVerdict],
    weights: WeightSettings,
    max_marks: float | None = None,
) -> tuple[float, float, str, float]:
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
    score_100 = min(weights.max_score, max(weights.min_score, total))
    grade = _letter_from_score(score_100, weights.grade_bands)
    ceiling = max_marks if max_marks and max_marks > 0 else weights.max_score
    awarded = score_100 / weights.max_score * ceiling if weights.max_score else 0.0
    return float(round_marks(awarded, ceiling)), confidence, grade, score_100
