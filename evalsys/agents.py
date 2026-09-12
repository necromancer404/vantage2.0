from __future__ import annotations

from evalsys.config import LLMSettings
from evalsys.llm.base import LLMError, LLMProvider
from evalsys.llm.json_util import extract_json_object, parse_verdict_payload
from evalsys.prompts import AGENT_SYSTEM_PROMPTS, build_user_prompt
from evalsys.schemas import AGENT_NAMES, AgentVerdict, EvaluationSample


class AgentRunner:
    def __init__(self, name: str, provider: LLMProvider, settings: LLMSettings) -> None:
        if name not in AGENT_NAMES:
            raise ValueError(f"Unknown agent '{name}'")
        self.name = name
        self.provider = provider
        self.settings = settings
        self.system_prompt = AGENT_SYSTEM_PROMPTS[name]

    def evaluate(self, sample: EvaluationSample) -> AgentVerdict:
        user_prompt = build_user_prompt(
            question=sample.question,
            submission=sample.submission,
            criterion=sample.criterion,
            ratings=sample.ratings,
        )
        try:
            response = self.provider.complete(system=self.system_prompt, user=user_prompt)
            parsed = parse_verdict_payload(extract_json_object(response.text))
            return AgentVerdict(
                name=self.name,
                score=parsed["score"],
                confidence=parsed["confidence"],
                summary=parsed["summary"],
                findings=parsed["findings"],
                grade=parsed["grade"],
                raw_text=response.text,
                model=response.model,
                provider=response.provider,
            )
        except (LLMError, ValueError, TypeError) as exc:
            return AgentVerdict(
                name=self.name,
                score=0.0,
                confidence=0.0,
                summary="Agent call failed.",
                findings=[],
                raw_text="",
                model=self.settings.model,
                provider=self.provider.provider_name,
                error=str(exc),
            )
