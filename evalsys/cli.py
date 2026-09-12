from __future__ import annotations

import argparse
from pathlib import Path

from evalsys.config import ROOT
from evalsys.csv_store import ResultStore
from evalsys.dataset import load_samples, sample_from_code_file
from evalsys.llm.factory import build_provider
from evalsys.orchestrator import EvaluationOrchestrator
from evalsys.presets import (
    describe_agents,
    list_mixes,
    list_presets,
    load_runtime_config,
    suite_names,
)
from evalsys.schemas import AGENT_NAMES, EvaluationSample


def _parse_agent_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use --agent name=preset, e.g. --agent style=groq")
    name, preset = value.split("=", 1)
    name = name.strip()
    preset = preset.strip()
    if name not in AGENT_NAMES:
        raise argparse.ArgumentTypeError(f"Unknown agent '{name}'. Known: {', '.join(AGENT_NAMES)}")
    if not preset:
        raise argparse.ArgumentTypeError("Preset name is empty")
    return name, preset


def _add_eval_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--xlsx", type=Path, default=ROOT / "test_30_each_ground_truth.xlsx")
    parser.add_argument("--sheet", default="all_samples")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "evaluations.csv")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM providers (no API calls)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--preset",
        default="",
        help="Use this model for ALL agents",
    )
    parser.add_argument(
        "--mix",
        default="",
        help="Use a named mix from config/models.yaml (different model per agent)",
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        type=_parse_agent_assignment,
        metavar="NAME=PRESET",
        help="Override one agent, e.g. --agent style=groq (repeatable)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent automated code evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="Evaluate rows from the ground-truth workbook")
    _add_eval_flags(batch)
    batch.add_argument(
        "--sweep",
        nargs="?",
        const="suite",
        default="",
        help="Run the test_suite (or comma-separated preset names) sequentially",
    )

    one = sub.add_parser("file", help="Evaluate a single submission file")
    one.add_argument("code", type=Path)
    one.add_argument("--question", required=True)
    one.add_argument("--criterion", default="")
    one.add_argument("--out", type=Path, default=ROOT / "results" / "evaluations.csv")
    one.add_argument("--mock", action="store_true")
    one.add_argument("--workers", type=int, default=5)
    one.add_argument("--preset", default="")
    one.add_argument("--mix", default="")
    one.add_argument(
        "--agent",
        action="append",
        default=[],
        type=_parse_agent_assignment,
        metavar="NAME=PRESET",
    )

    listed = sub.add_parser("presets", help="List judge model presets and mixes")
    listed.add_argument("--models", type=Path, default=ROOT / "config" / "models.yaml")

    ping = sub.add_parser("ping", help="Send one tiny request to a preset or to each agent in a mix")
    ping.add_argument("--preset", default="")
    ping.add_argument("--mix", default="")
    ping.add_argument("--mock", action="store_true")
    return parser


def _assignments(agent_flags: list[tuple[str, str]] | None) -> dict[str, str]:
    return dict(agent_flags or [])


def _print_record(record) -> None:
    tag = f"[{record.preset}] " if record.preset else ""
    print(f"{tag}{record.sample.sample_id}  total={record.total_score:.2f}  "
          f"conf={record.overall_confidence:.3f}  grade={record.predicted_grade}")
    for name, verdict in record.verdicts.items():
        err = f"  ERROR={verdict.error}" if verdict.error else ""
        print(f"  {name:13} {verdict.model:32} score={verdict.score:6.2f}  conf={verdict.confidence:.3f}{err}")


def _resolve_sweep(value: str) -> list[str]:
    if not value or value == "suite":
        return suite_names()
    return [item.strip() for item in value.split(",") if item.strip()]


def _run(
    samples: list[EvaluationSample],
    out: Path,
    mock: bool,
    workers: int,
    preset: str = "",
    mix: str = "",
    agent_assignments: dict[str, str] | None = None,
) -> int:
    config = load_runtime_config(mix=mix, preset=preset, agent_assignments=agent_assignments, mock=mock)
    print("Agent models:")
    for line in describe_agents(config):
        print(f"  {line}")
    orchestrator = EvaluationOrchestrator(config, max_workers=workers)
    store = ResultStore(out)
    for sample in samples:
        record = orchestrator.evaluate(sample)
        store.append(record)
        _print_record(record)
    print(f"Appended {len(samples)} row(s) to {out}")
    return 0


def _cmd_presets(path: Path) -> int:
    presets = list_presets(path)
    width = max((len(name) for name in presets), default=8)
    print(f"{'PRESET'.ljust(width)}  PROVIDER     MODEL")
    for name, raw in presets.items():
        print(f"{name.ljust(width)}  {str(raw.get('provider', '')):12} {raw.get('model')}")
    mixes = list_mixes(path)
    print("\nMIXES")
    for name, raw in mixes.items():
        print(f"  {name}: {raw.get('label', '')}")
        agents = raw.get("agents") or {}
        for agent in AGENT_NAMES:
            print(f"    {agent:13} {agents.get(agent)}")
    print("\nTest suite:", ", ".join(suite_names(path)))
    return 0


def _cmd_ping(preset: str, mix: str, mock: bool) -> int:
    config = load_runtime_config(mix=mix, preset=preset, mock=mock)
    targets = {"all": config.llm_default} if preset else config.llm_agents
    for name, settings in targets.items():
        provider = build_provider(settings)
        response = provider.complete(
            system="Reply with compact JSON only.",
            user='Return {"ok": true, "model": "' + settings.model + '"}',
        )
        print(f"{name}: provider={settings.provider} model={settings.model}")
        print(response.text[:500])
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "presets":
        return _cmd_presets(args.models)
    if args.command == "ping":
        try:
            return _cmd_ping(args.preset, args.mix, args.mock)
        except Exception as extra:
            print(f"Ping failed: {extra}")
            return 1
    if args.command == "batch":
        samples = load_samples(args.xlsx, sheet=args.sheet, limit=args.limit, offset=args.offset)
        if not samples:
            print("No samples found.")
            return 1
        if args.sweep:
            failed = 0
            for name in _resolve_sweep(args.sweep):
                print(f"\n=== preset {name} ===")
                try:
                    _run(samples, args.out, args.mock, args.workers, preset=name)
                except Exception as extra:
                    failed += 1
                    print(f"Preset {name} failed: {extra}")
            return 1 if failed else 0
        return _run(
            samples,
            args.out,
            args.mock,
            args.workers,
            preset=args.preset,
            mix=args.mix,
            agent_assignments=_assignments(args.agent),
        )
    if args.command == "file":
        sample = sample_from_code_file(args.code, args.question, args.criterion)
        return _run(
            [sample],
            args.out,
            args.mock,
            args.workers,
            preset=args.preset,
            mix=args.mix,
            agent_assignments=_assignments(args.agent),
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
