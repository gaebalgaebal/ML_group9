"""Command-line interface for the recommendation project.

Entry point: `python -m ml_group9 <command> --profile {sample,medium,full}` with the commands prepare,
prepare-types, train, evaluate, run, inspect-cj, tune-re, eval-re, tune-hs, eval-hs, tune-hier,
eval-hier and analyze-hstar. Each command writes its own files and prints its result as JSON.
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import complete_journey_department_summary, prepare
from .health_subgroup import eval_hs, tune_hs
from .hier_pipeline import eval_hier, tune_hier
from .hstar import analyze_hstar
from .ingredient_types import prepare_types
from .pipeline import evaluate, run, train
from .re_pipeline import eval_re, tune_re


PROFILES = ["sample", "medium", "full"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid next-basket recommendation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Normalize raw archives and create temporal splits")
    prepare_parser.add_argument("--dataset", choices=["instacart", "complete_journey", "all"], default="all")
    prepare_parser.add_argument("--profile", choices=PROFILES, default="sample")

    train_parser = subparsers.add_parser("train", help="Train the Instacart logistic ranker")
    train_parser.add_argument("--profile", choices=PROFILES, default="sample")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the frozen ranker")
    evaluate_parser.add_argument("--dataset", choices=["instacart", "complete_journey"], required=True)
    evaluate_parser.add_argument("--profile", choices=PROFILES, default="sample")

    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--profile", choices=PROFILES, default="sample")

    inspect_parser = subparsers.add_parser(
        "inspect-cj", help="Print Complete Journey DEPARTMENT shares to choose excluded departments"
    )
    inspect_parser.add_argument("--profile", choices=PROFILES, default="sample")

    tune_parser = subparsers.add_parser(
        "tune-re", help="Tune repeat-explore on the Instacart fit split (and the CJ fit split for refit)"
    )
    tune_parser.add_argument("--profile", choices=PROFILES, default="sample")

    eval_re_parser = subparsers.add_parser("eval-re", help="Evaluate repeat-explore and baselines on an eval split")
    eval_re_parser.add_argument("--dataset", choices=["instacart", "complete_journey"], required=True)
    eval_re_parser.add_argument("--profile", choices=PROFILES, default="sample")

    tune_hs_parser = subparsers.add_parser(
        "tune-hs", help="Select beta for health-attribute subgroups on the Instacart fit split"
    )
    tune_hs_parser.add_argument("--profile", choices=PROFILES, default="sample")

    eval_hs_parser = subparsers.add_parser(
        "eval-hs", help="Evaluate health-attribute subgroups on the Instacart eval split"
    )
    eval_hs_parser.add_argument("--profile", choices=PROFILES, default="sample")

    types_parser = subparsers.add_parser(
        "prepare-types", help="Import the ingredient-type mapping and report type-level attribute categories"
    )
    types_parser.add_argument("--profile", choices=PROFILES, default="sample")

    tune_hier_parser = subparsers.add_parser(
        "tune-hier", help="Tune the hierarchical recommender on the fit splits (food scope)"
    )
    tune_hier_parser.add_argument("--profile", choices=PROFILES, default="sample")

    eval_hier_parser = subparsers.add_parser(
        "eval-hier", help="Evaluate the hierarchical recommender, ablations and baselines on an eval split"
    )
    eval_hier_parser.add_argument("--dataset", choices=["instacart", "complete_journey"], required=True)
    eval_hier_parser.add_argument("--profile", choices=PROFILES, default="sample")

    hstar_parser = subparsers.add_parser(
        "analyze-hstar", help="Descriptive H* analysis (stability, extension, concentration) on the Instacart eval split"
    )
    hstar_parser.add_argument("--profile", choices=PROFILES, default="sample")
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
    elif args.command == "tune-re":
        result = tune_re(config)
    elif args.command == "eval-re":
        result = eval_re(config, args.dataset)
    elif args.command == "tune-hs":
        result = tune_hs(config)
    elif args.command == "eval-hs":
        result = eval_hs(config)
    elif args.command == "prepare-types":
        result = prepare_types(config)
    elif args.command == "tune-hier":
        result = tune_hier(config)
    elif args.command == "eval-hier":
        result = eval_hier(config, args.dataset)
    elif args.command == "analyze-hstar":
        result = analyze_hstar(config)
    else:
        result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
