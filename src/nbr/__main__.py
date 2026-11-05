"""Command line entry point: ``python -m nbr check --profile P``."""

from __future__ import annotations

import argparse

import pandas as pd

from nbr.config import load_config
from nbr.data import load_split, split_target


def describe_split(history: pd.DataFrame, target: pd.DataFrame) -> dict[str, int]:
    """Row/user counts for one split, including repeat/new target user counts."""
    repeat_target, new_target = split_target(history, target)
    return {
        "users": int(target["user_id"].nunique()),
        "history_rows": int(len(history)),
        "target_rows": int(len(target)),
        "users_with_repeat_target": int(repeat_target["user_id"].nunique()),
        "users_with_new_target": int(new_target["user_id"].nunique()),
    }


def check(profile: str) -> None:
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
    parser = argparse.ArgumentParser(prog="python -m nbr")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="print split statistics per dataset")
    check_parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        check(args.profile)


if __name__ == "__main__":
    main()
