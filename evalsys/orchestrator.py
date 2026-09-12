from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from evalsys.agents import AgentRunner
from evalsys.config import AppConfig
from evalsys.llm.factory import build_provider
from evalsys.schemas import AGENT_NAMES, AgentVerdict, EvaluationRecord, EvaluationSample
from evalsys.scoring import combine_verdicts


class EvaluationOrchestrator:
    def __init__(self, config: AppConfig, max_workers: int = 5) -> None:
        self.config = config
        self.max_workers = max_workers
        self.agents = {
            name: AgentRunner(name, build_provider(config.llm_agents[name]), config.llm_agents[name])
            for name in AGENT_NAMES
        }

    def evaluate(self, sample: EvaluationSample) -> EvaluationRecord:
        verdicts: dict[str, AgentVerdict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(agent.evaluate, sample): name for name, agent in self.agents.items()}
            for future in as_completed(futures):
                name = futures[future]
                verdicts[name] = future.result()

        total, confidence, grade = combine_verdicts(verdicts, self.config.weights)
        return EvaluationRecord(
            sample=sample,
            verdicts=verdicts,
            total_score=total,
            overall_confidence=confidence,
            predicted_grade=grade,
            preset=self.config.preset,
        )
