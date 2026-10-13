from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from evalsys.schemas import EvaluationSample

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_FRACTION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_EXCEL_DAY_MONTH = re.compile(
    r"^\s*(\d{1,2})\s*[-/]\s*([A-Za-z]{3,9})\s*$"
)


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text in {"0", "0.0", "nan", "NaN"}:
        return ""
    return text


def _row_get(row: pd.Series, *names: str):
    lookup = {str(key).strip().lower(): key for key in row.index}
    for name in names:
        key = lookup.get(name.strip().lower())
        if key is not None:
            return row.get(key)
    return None


def parse_marks(value) -> tuple[float | None, float | None]:
    """Parse awarded and maximum marks from dataset cells.

    Source files often store ``awarded/max`` (e.g. ``3/4``). Excel commonly
    rewrites those as dates (``03-Apr`` or a datetime), where the day is the
    awarded score and the month number is the maximum.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    if isinstance(value, pd.Timestamp):
        return float(value.day), float(value.month)
    if hasattr(value, "day") and hasattr(value, "month") and not isinstance(value, (int, float, str)):
        try:
            return float(value.day), float(value.month)
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)) and not pd.isna(value):
        number = float(value)
        if number <= 0:
            return None, None
        return None, number

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None, None

    match = _FRACTION.match(text)
    if match:
        awarded = float(match.group(1))
        maximum = float(match.group(2))
        return awarded, maximum if maximum > 0 else None

    match = _EXCEL_DAY_MONTH.match(text)
    if match:
        awarded = float(match.group(1))
        month = _MONTHS.get(match.group(2).lower()[:3])
        if month:
            return awarded, float(month)

    try:
        parsed = pd.to_datetime(text, dayfirst=True, errors="raise")
        if pd.notna(parsed):
            return float(parsed.day), float(parsed.month)
    except (ValueError, TypeError, OverflowError):
        pass

    try:
        number = float(text)
    except ValueError:
        return None, None
    if number <= 0:
        return None, None
    return None, number


def _read_frame(path: Path, sheet: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported dataset type '{path.suffix}'. Use .csv or .xlsx")


def load_samples(
    path: Path,
    sheet: str = "all_samples",
    limit: int | None = None,
    offset: int = 0,
    default_max_marks: float = 100.0,
) -> list[EvaluationSample]:
    frame = _read_frame(path, sheet)
    samples: list[EvaluationSample] = []
    sliced = frame.iloc[offset : offset + limit if limit is not None else None]
    source = path.stem
    for index, row in sliced.iterrows():
        submission = _clean(_row_get(row, "submission", "solution", "code"))
        question = _clean(_row_get(row, "question", "problem", "problem statement"))
        if not question and not submission:
            continue
        ratings = {
            "A": _clean(_row_get(row, "Rating A")),
            "B": _clean(_row_get(row, "Rating B")),
            "C": _clean(_row_get(row, "Rating C")),
            "D": _clean(_row_get(row, "Rating D")),
        }
        awarded, maximum = parse_marks(_row_get(row, "total marks", "total_marks", "marks", "max marks"))
        max_marks = maximum if maximum and maximum > 0 else default_max_marks
        category = _clean(_row_get(row, "category"))
        rubric = _clean(_row_get(row, "rubric", "rubrics", "marking scheme", "marking_scheme"))
        ground_truth = _clean(_row_get(row, "Ground Truth", "ground_truth"))
        if not ground_truth and awarded is not None:
            ground_truth = f"{awarded:g}/{max_marks:g}"
        samples.append(
            EvaluationSample(
                sample_id=f"{source}:{index}",
                question=question,
                submission=submission,
                criterion=_clean(_row_get(row, "criterion")),
                ratings=ratings,
                ground_truth=ground_truth,
                category=category,
                max_marks=max_marks,
                awarded_marks=awarded,
                rubric=rubric,
            )
        )
    return samples


def sample_from_code_file(
    code_path: Path,
    question: str,
    criterion: str = "",
    max_marks: float = 100.0,
    rubric: str = "",
) -> EvaluationSample:
    return EvaluationSample(
        sample_id=code_path.name,
        question=question,
        submission=code_path.read_text(encoding="utf-8", errors="replace"),
        criterion=criterion,
        max_marks=max_marks,
        rubric=rubric,
    )
