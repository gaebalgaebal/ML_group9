"""Command line entry point.

    python -m nbr check --profile P
    python -m nbr tune --profile P --dataset D
    python -m nbr evaluate --profile P --dataset D [--transfer-from instacart] [--final] [--no-itemcf] [--no-lr] [--no-integration] [--skip-l3] [--policy]

tune also fits the W6 logistic-regression repeat ranker (stored under
artifacts/<profile>/<dataset>_lr_repeat.joblib) and runs the W8 user-CF / slot
selection; evaluate loads them unless --no-lr / --no-integration. --policy (W17) adds the
PREREG v1.4 exploration-policy block (nbr.policy) to the eval JSON; it needs both.
With lr_repeat.itemcf_feature: true in the profile (PREREG v1.5, W19) tune also fits R2'
(artifacts/<profile>/<dataset>_lr_repeat_icf.joblib) and selects r of C4, and evaluate adds the
"v15" block (nbr.v15: H14, H15); no flag is needed.

--final is the only way to read the Instacart eval split of a role == "final"
profile (holdout); without it the holdout guard raises PermissionError.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from nbr.config import load_config
from nbr.data import load_split, split_target


def describe_split(history: pd.DataFrame, target: pd.DataFrame) -> dict[str, int]:
    """Row/user counts of one split from its ``history`` and ``target`` frames, including repeat/new target user counts."""
    repeat_target, new_target = split_target(history, target)
    return {
        "users": int(target["user_id"].nunique()),
        "history_rows": int(len(history)),
        "target_rows": int(len(target)),
        "users_with_repeat_target": int(repeat_target["user_id"].nunique()),
        "users_with_new_target": int(new_target["user_id"].nunique()),
    }


def check(profile: str) -> None:
    """Print row/user counts of every dataset and split of ``profile``; the holdout guard skips the final eval split."""
    config = load_config(profile)
    print(f"profile={profile} role={config.get('role')} data_dir={config['data_dir']}")
    for dataset in config["datasets"]:
        for split in ("fit", "eval"):
            try:
                history, target = load_split(config, dataset, split)
            except PermissionError:
                print(f"{dataset}/{split}: skipped (holdout guard)")
                continue
            stats = describe_split(history, target)
            summary = " ".join(f"{key}={value}" for key, value in stats.items())
            print(f"{dataset}/{split}: {summary}")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default: sys.argv) and run the check / tune / evaluate command."""
    parser = argparse.ArgumentParser(prog="python -m nbr")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="print split statistics per dataset")
    check_parser.add_argument("--profile", required=True)
    tune_parser = sub.add_parser("tune", help="fit-split selection -> reports/metrics/<profile>_<dataset>_tuning.json")
    tune_parser.add_argument("--profile", required=True)
    tune_parser.add_argument("--dataset", required=True)
    eval_parser = sub.add_parser("evaluate", help="eval-split evaluation -> reports/metrics/<profile>_<dataset>_eval.json")
    eval_parser.add_argument("--profile", required=True)
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--transfer-from", default=None, help="use the selections of another dataset")
    eval_parser.add_argument("--final", action="store_true", help="allow reading the holdout eval split")
    eval_parser.add_argument("--no-itemcf", action="store_true", help="skip the item-CF auxiliary models")
    eval_parser.add_argument("--no-lr", action="store_true", help="skip the logistic-regression repeat ranker (H6, H7)")
    eval_parser.add_argument("--no-integration", action="store_true", help="skip user-CF and the W8 combinations (H8, H9)")
    eval_parser.add_argument("--skip-l3", action="store_true", help="skip the explore_L3 list and its secondary comparison")
    eval_parser.add_argument("--policy", action="store_true", help="add the PREREG v1.4 exploration-policy block (H11-H13)")
    args = parser.parse_args(argv)

    if args.command == "check":
        check(args.profile)
        return
    from nbr.pipeline import run_evaluate, run_tune  # imported lazily: heavy modules

    config = load_config(args.profile)
    t0 = time.perf_counter()
    if args.command == "tune":
        path = run_tune(config, args.dataset)
    else:
        path = run_evaluate(
            config, args.dataset, transfer_from=args.transfer_from,
            allow_holdout_eval=args.final, include_itemcf=not args.no_itemcf, include_lr=not args.no_lr,
            include_integration=not args.no_integration, skip_l3=args.skip_l3, include_policy=args.policy,
        )
    print(f"{args.command} {args.profile}/{args.dataset}: wrote {path} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
