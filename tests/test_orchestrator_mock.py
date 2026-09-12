from pathlib import Path

from evalsys.config import load_config
from evalsys.csv_store import ResultStore
from evalsys.orchestrator import EvaluationOrchestrator
from evalsys.schemas import EvaluationSample


def test_mock_pipeline_appends_csv(tmp_path: Path):
    config = load_config()
    for settings in config.llm_agents.values():
        settings.provider = "mock"
    orchestrator = EvaluationOrchestrator(config)
    sample = EvaluationSample(
        sample_id="unit-1",
        question="Read n and print n*n.",
        submission="int main(){int n; cin>>n; cout<<n*n;}",
        criterion="Overall program correctness",
        ground_truth="B",
    )
    record = orchestrator.evaluate(sample)
    assert record.total_score > 0
    assert set(record.verdicts) == {
        "correctness",
        "style",
        "counteragent",
        "edge_cases",
        "complexity",
    }
    csv_path = tmp_path / "out.csv"
    ResultStore(csv_path).append(record)
    text = csv_path.read_text(encoding="utf-8")
    assert "total_score" in text
    assert "correctness_score" in text
    assert "overall_confidence" in text
    assert "preset" in text
