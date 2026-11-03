"""Command-line interface for the recommendation project."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import complete_journey_department_summary, prepare
from .pipeline import evaluate, run, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid next-basket recommendation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Normalize raw archives and create temporal splits")
    prepare_parser.add_argument("--dataset", choices=["instacart", "complete_journey", "all"], default="all")
    prepare_parser.add_argument("--profile", choices=["sample", "full"], default="sample")

    train_parser = subparsers.add_parser("train", help="Train the Instacart logistic ranker")
    train_parser.add_argument("--profile", choices=["sample", "full"], default="sample")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the frozen ranker")
    evaluate_parser.add_argument("--dataset", choices=["instacart", "complete_journey"], required=True)
    evaluate_parser.add_argument("--profile", choices=["sample", "full"], default="sample")

    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--profile", choices=["sample", "full"], default="sample")

    inspect_parser = subparsers.add_parser(
        "inspect-cj", help="Print Complete Journey DEPARTMENT shares to choose excluded departments"
    )
    inspect_parser.add_argument("--profile", choices=["sample", "full"], default="sample")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.profile)
    if args.command == "prepare":
        result = prepare(config, args.dataset)
    elif args.command == "train":
        result = train(config)
    elif args.command == "evaluate":
        result = evaluate(config, args.dataset)
    elif args.command == "inspect-cj":
        result = complete_journey_department_summary(config)
    else:
        result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
