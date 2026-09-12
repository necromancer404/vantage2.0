from evalsys.scoring import combine_verdicts
from evalsys.schemas import AgentVerdict
from evalsys.config import WeightSettings


def _verdict(name: str, score: float, confidence: float = 1.0) -> AgentVerdict:
    return AgentVerdict(name=name, score=score, confidence=confidence, summary="", findings=[])


def test_weighted_total():
    weights = WeightSettings(
        agents={
            "correctness": 0.35,
            "style": 0.10,
            "counteragent": 0.20,
            "edge_cases": 0.20,
            "complexity": 0.15,
        },
        grade_bands={"A": 85, "B": 70, "C": 50, "D": 0},
        disagreement_spread_penalty=0.0,
    )
    verdicts = {
        "correctness": _verdict("correctness", 100),
        "style": _verdict("style", 50),
        "counteragent": _verdict("counteragent", 80),
        "edge_cases": _verdict("edge_cases", 70),
        "complexity": _verdict("complexity", 60),
    }
    total, confidence, grade = combine_verdicts(verdicts, weights)
    expected = 0.35 * 100 + 0.10 * 50 + 0.20 * 80 + 0.20 * 70 + 0.15 * 60
    assert abs(total - expected) < 1e-6
    assert abs(confidence - 1.0) < 1e-6
    assert grade == "B"
