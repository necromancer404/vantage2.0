from __future__ import annotations

import csv
import json
from pathlib import Path

from evalsys.schemas import AGENT_NAMES, EvaluationRecord

CSV_FIELDS = [
    "timestamp_utc",
    "sample_id",
    "preset",
    "judge_model",
    "judge_provider",
    "criterion",
    "ground_truth",
    "predicted_grade",
    "total_score",
    "overall_confidence",
    *[
        field
        for name in AGENT_NAMES
        for field in (
            f"{name}_score",
            f"{name}_confidence",
            f"{name}_grade",
            f"{name}_summary",
            f"{name}_findings",
            f"{name}_model",
            f"{name}_provider",
            f"{name}_error",
        )
    ],
    "question",
    "submission",
]


class ResultStore:
    def __init__(self, csv_path: Path, jsonl_path: Path | None = None) -> None:
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path or csv_path.with_suffix(".jsonl")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> None:
        exists = self.csv_path.exists()
        row = record.csv_row()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        payload = {
            "timestamp_utc": record.created_at.isoformat(),
            "sample_id": record.sample.sample_id,
            "total_score": record.total_score,
            "overall_confidence": record.overall_confidence,
            "predicted_grade": record.predicted_grade,
            "preset": record.preset,
            "ground_truth": record.sample.ground_truth,
            "verdicts": {
                name: {
                    "score": verdict.score,
                    "confidence": verdict.confidence,
                    "grade": verdict.grade,
                    "summary": verdict.summary,
                    "findings": verdict.findings,
                    "model": verdict.model,
                    "provider": verdict.provider,
                    "error": verdict.error,
                    "raw_text": verdict.raw_text,
                }
                for name, verdict in record.verdicts.items()
            },
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
