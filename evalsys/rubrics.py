from __future__ import annotations

import re

from evalsys.schemas import AGENT_NAMES

_KEYWORDS: dict[str, tuple[re.Pattern[str], ...]] = {
    "style": (
        re.compile(r"\bstyle\b"),
        re.compile(r"\bnam(?:e|ing)\b"),
        re.compile(r"\bcomment"),
        re.compile(r"\bdocument"),
        re.compile(r"\breadab"),
        re.compile(r"\bformat"),
        re.compile(r"\bindent"),
        re.compile(r"\bmaintainab"),
        re.compile(r"\bmagic number"),
        re.compile(r"\bdead code"),
        re.compile(r"\bwhitespace"),
        re.compile(r"\bclean code"),
        re.compile(r"\bconvention"),
        re.compile(r"\bidentifier"),
        re.compile(r"\bcamelcase"),
        re.compile(r"\bsnake_case"),
    ),
    "complexity": (
        re.compile(r"time complexity"),
        re.compile(r"space complexity"),
        re.compile(r"auxiliary space"),
        re.compile(r"\bcomplexity\b"),
        re.compile(r"\bo\s*\("),
        re.compile(r"\befficient"),
        re.compile(r"\befficiency"),
        re.compile(r"\btle\b"),
        re.compile(r"\btimeout"),
        re.compile(r"\basymptotic"),
        re.compile(r"\bbig-?o\b"),
        re.compile(r"\bperformance\b"),
        re.compile(r"\boptimal(?:ity)?\b"),
    ),
    "edge_cases": (
        re.compile(r"edge case"),
        re.compile(r"\bboundar"),
        re.compile(r"\bcorner case"),
        re.compile(r"empty (?:string|array|list|input|case)"),
        re.compile(r"\boverflow"),
        re.compile(r"\bunderflow"),
        re.compile(r"\bspecial case"),
        re.compile(r"\bsingleton"),
        re.compile(r"\boff-by-one"),
        re.compile(r"\bprecision"),
        re.compile(r"invalid input"),
        re.compile(r"n\s*=\s*0"),
        re.compile(r"n\s*=\s*1"),
    ),
    "counteragent": (
        re.compile(r"\bhardcod"),
        re.compile(r"\bcounterexample"),
        re.compile(r"\btest case"),
        re.compile(r"\badversar"),
        re.compile(r"\bcheat"),
        re.compile(r"wrong answer"),
        re.compile(r"fails for"),
        re.compile(r"does not handle"),
        re.compile(r"\bbypass"),
    ),
    "correctness": (
        re.compile(r"\bsolution\b"),
        re.compile(r"\balgorithm\b"),
        re.compile(r"\breturn\b"),
        re.compile(r"\boutput\b"),
        re.compile(r"\bcorrect"),
        re.compile(r"\blogic\b"),
        re.compile(r"\binitialize"),
        re.compile(r"\biterate"),
        re.compile(r"\[\s*\d+\s*marks?\s*\]"),
    ),
}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def classify_rubric_line(line: str) -> set[str]:
    text = line.strip().lower()
    if not text:
        return set()
    matched = {
        agent
        for agent, patterns in _KEYWORDS.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    return matched


def _line_matches_agent(agent: str, matched: set[str]) -> bool:
    if agent == "correctness":
        specialists = matched - {"correctness", "counteragent"}
        return (not matched) or ("correctness" in matched) or (not specialists)
    return agent in matched


def rubric_for_agent(agent: str, rubric: str) -> str:
    if agent not in AGENT_NAMES:
        raise ValueError(f"Unknown agent '{agent}'")
    text = (rubric or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    keep = [False] * len(lines)
    matched_any = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if _line_matches_agent(agent, classify_rubric_line(line)):
            keep[index] = True
            matched_any = True
            indent = _indent(line)
            for prior in range(index - 1, -1, -1):
                if not lines[prior].strip():
                    continue
                prior_indent = _indent(lines[prior])
                if prior_indent < indent:
                    keep[prior] = True
                    indent = prior_indent
                if prior_indent == 0:
                    keep[prior] = True
                    break

    if not matched_any:
        return ""

    selected: list[str] = []
    for index, line in enumerate(lines):
        if keep[index]:
            selected.append(line.rstrip())
            continue
        if selected and line.strip() == "" and (index + 1 < len(lines) and keep[index + 1]):
            selected.append("")
    return "\n".join(selected).strip()
