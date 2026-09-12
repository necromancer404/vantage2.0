from __future__ import annotations

from pathlib import Path

import pandas as pd

from evalsys.schemas import EvaluationSample


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text in {"0", "0.0", "nan", "NaN"}:
        return ""
    return text


def load_samples(
    path: Path,
    sheet: str = "all_samples",
    limit: int | None = None,
    offset: int = 0,
) -> list[EvaluationSample]:
    frame = pd.read_excel(path, sheet_name=sheet)
    samples: list[EvaluationSample] = []
    sliced = frame.iloc[offset : offset + limit if limit is not None else None]
    for index, row in sliced.iterrows():
        submission = _clean(row.get("Submission"))
        question = _clean(row.get("Question"))
        if not question and not submission:
            continue
        ratings = {
            "A": _clean(row.get("Rating A")),
            "B": _clean(row.get("Rating B")),
            "C": _clean(row.get("Rating C")),
            "D": _clean(row.get("Rating D")),
        }
        samples.append(
            EvaluationSample(
                sample_id=f"{sheet}:{index}",
                question=question,
                submission=submission,
                criterion=_clean(row.get("criterion")),
                ratings=ratings,
                ground_truth=_clean(row.get("Ground Truth")),
            )
        )
    return samples


def sample_from_code_file(
    code_path: Path,
    question: str,
    criterion: str = "",
) -> EvaluationSample:
    return EvaluationSample(
        sample_id=code_path.name,
        question=question,
        submission=code_path.read_text(encoding="utf-8", errors="replace"),
        criterion=criterion,
    )
