from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


AGENT_NAMES = (
    "correctness",
    "style",
    "counteragent",
    "edge_cases",
    "complexity",
)


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    question: str
    submission: str
    criterion: str = ""
    ratings: dict[str, str] = field(default_factory=dict)
    ground_truth: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentVerdict:
    name: str
    score: float
    confidence: float
    summary: str
    findings: list[str]
    grade: str = ""
    raw_text: str = ""
    model: str = ""
    provider: str = ""
    error: str = ""

    def as_csv_fields(self) -> dict[str, Any]:
        return {
            f"{self.name}_score": round(self.score, 4),
            f"{self.name}_confidence": round(self.confidence, 4),
            f"{self.name}_grade": self.grade,
            f"{self.name}_summary": self.summary,
            f"{self.name}_findings": " | ".join(self.findings),
            f"{self.name}_model": self.model,
            f"{self.name}_provider": self.provider,
            f"{self.name}_error": self.error,
        }


@dataclass
class EvaluationRecord:
    sample: EvaluationSample
    verdicts: dict[str, AgentVerdict]
    total_score: float
    overall_confidence: float
    predicted_grade: str
    preset: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def csv_row(self) -> dict[str, Any]:
        models = sorted({v.model for v in self.verdicts.values() if v.model})
        providers = sorted({v.provider for v in self.verdicts.values() if v.provider})
        row: dict[str, Any] = {
            "timestamp_utc": self.created_at.isoformat(),
            "sample_id": self.sample.sample_id,
            "preset": self.preset,
            "judge_model": " | ".join(models),
            "judge_provider": " | ".join(providers),
            "criterion": self.sample.criterion,
            "ground_truth": self.sample.ground_truth,
            "predicted_grade": self.predicted_grade,
            "total_score": round(self.total_score, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "question": self.sample.question,
            "submission": self.sample.submission,
        }
        for name in AGENT_NAMES:
            verdict = self.verdicts.get(name)
            if verdict:
                row.update(verdict.as_csv_fields())
        return row
