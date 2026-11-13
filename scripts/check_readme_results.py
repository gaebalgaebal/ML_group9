# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Render the README result tables from the master hypotheses table (T1) and check
          README.md against T1 (W11b).
          Every number in the README result tables comes from
          reports/results/tables/T1_master_hypotheses.csv, written by scripts/make_results.py;
          nothing is typed in by hand. The tables, their rows and the rounding are defined
          here (TABLES).
Inputs  : reports/results/tables/T1_master_hypotheses.csv (--t1) and README.md (--readme).
          Only T1 rows with ``current = true`` are used.
          In README.md each table follows a marker line ``<!-- results-table: NAME -->``.
Outputs : Check mode: the check reads the Markdown table after each marker and compares it
          cell by cell with the rendered table (missing or extra rows count as mismatches);
          every mismatch and the mismatch count are printed; exit 1 on a mismatch.
          Render mode: the tables (Markdown) that belong in README.md are printed.
Usage   :
    python scripts/check_readme_results.py
        (check: every cell of the README result tables vs T1; exit 1 on a mismatch)
    python scripts/check_readme_results.py --render
        (print the tables (Markdown) that belong in README.md)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# paths and labels
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
T1_PATH = REPO_ROOT / "reports" / "results" / "tables" / "T1_master_hypotheses.csv"
README_PATH = REPO_ROOT / "README.md"
MARKER = "<!-- results-table: {name} -->"

H4 = "confirmatory holdout4"
H3 = "confirmatory holdout3"
H2 = "confirmatory holdout2"
H1 = "confirmatory holdout"
INSTACART4 = "Instacart holdout4"
INSTACART3 = "Instacart holdout3"
INSTACART2 = "Instacart holdout2"
CJ_REFIT = "Complete Journey refit"
CJ_TRANSFER = "Complete Journey, Instacart selections transferred"


# ---------------------------------------------------------------------------
# table definitions
# ---------------------------------------------------------------------------
def row(label: str, dataset_text: str, stage: str, id_: str, dataset: str = "instacart", transfer: str = "") -> dict:
    '''
    Row specification of one README table row: the README texts ``label`` and
    ``dataset_text`` and the T1 keys ``stage``, ``id_``, ``dataset`` and ``transfer``.
    '''
    return {"label": label, "dataset_text": dataset_text, "stage": stage, "id": id_, "dataset": dataset, "transfer_from": transfer}


# Table name -> (format, rows).
# Format "v15": 95 % and 97.5 % CIs with both decisions (rows of the PREREG v1.5 run, holdout4).
# Every block of that run except its policy blocks carries the 97.5 % level.
# Format "v14": 95 % and 98.33 % CIs with both decisions (policy rows of PREREG v1.4 and v1.5.1).
# Format "v13": 95 % and 99.375 % CIs with both decisions.
# Format "v12": 95 % CI and its decision.
TABLES: dict[str, tuple[str, list[dict]]] = {
    "holdout4_primary": ("v15", [
        row("H14 (NDCG@10, non-inferiority): C4 vs C3", INSTACART4, H4, "H14_ndcg"),
        row("H15 (NDCG@10): C4 vs personal frequency", INSTACART4, H4, "H15_ndcg"),
    ]),
    "holdout4_secondary": ("v15", [
        row("R2' vs R2 (repeat Recall@10)", INSTACART4, H4, "v15:repeat_lri_vs_repeat_lr_repeat_recall"),
        row("R2' vs R1 (repeat Recall@10)", INSTACART4, H4, "v15:repeat_lri_vs_repeat_star_repeat_recall"),
        row("C4 vs C3 (Recall@10)", INSTACART4, H4, "v15:final_lri_ucf_vs_final_lr_ucf_recall"),
    ]),
    "holdout4_replication": ("v15", [
        row("H6 (repeat Recall@10)", INSTACART4, H4, "H6_repeat_recall"),
        row("H7' (NDCG@10)", INSTACART4, H4, "H7prime_ndcg"),
        row("H7' (Recall@10, reported)", INSTACART4, H4, "H7prime_recall"),
        row("H8 (new-item Recall@10)", INSTACART4, H4, "H8_new_recall"),
    ]),
    "holdout4_complete_journey": ("v15", [
        row("H14 (NDCG@10, non-inferiority): C4 vs C3", CJ_REFIT, H4, "H14_ndcg", "complete_journey"),
        row("H15 (NDCG@10): C4 vs personal frequency", CJ_REFIT, H4, "H15_ndcg", "complete_journey"),
        row("R2' vs R2 (repeat Recall@10)", CJ_REFIT, H4, "v15:repeat_lri_vs_repeat_lr_repeat_recall", "complete_journey"),
        row("H14 (NDCG@10, non-inferiority): C4 vs C3", CJ_TRANSFER, H4, "H14_ndcg", "complete_journey", "instacart"),
        row("H15 (NDCG@10): C4 vs personal frequency", CJ_TRANSFER, H4, "H15_ndcg", "complete_journey", "instacart"),
        row("R2' vs R2 (repeat Recall@10)", CJ_TRANSFER, H4, "v15:repeat_lri_vs_repeat_lr_repeat_recall", "complete_journey", "instacart"),
    ]),
    "holdout4_policy": ("v14", [
        row("H11 (NDCG@10, non-inferiority): P_T vs C3", INSTACART3, H3, "H11_ndcg"),
        row("H12 (new-item hit share@10): P_T vs C3", INSTACART3, H3, "H12_new_hit_share"),
        row("H13 (new-item hit share@10): S vs flat popularity", INSTACART3, H3, "H13_new_hit_share"),
        row("as H11 (NDCG@10): P_T(C4) vs C4", INSTACART4, H4, "policy_c4:H11_ndcg"),
        row("as H12 (new-item hit share@10): P_T(C4) vs C4", INSTACART4, H4, "policy_c4:H12_new_hit_share"),
        row("as H13 (new-item hit share@10): S vs flat popularity", INSTACART4, H4, "policy_c4:H13_new_hit_share"),
        row("reference: P_T vs C3 (NDCG@10)", INSTACART4, H4, "H11_ndcg"),
        row("reference: P_T vs C3 (new-item hit share@10)", INSTACART4, H4, "H12_new_hit_share"),
    ]),
    "holdout3_primary": ("v14", [
        row("H11 (NDCG@10, non-inferiority)", INSTACART3, H3, "H11_ndcg"),
        row("H12 (new-item hit share@10)", INSTACART3, H3, "H12_new_hit_share"),
        row("H13 (new-item hit share@10, S vs flat popularity)", INSTACART3, H3, "H13_new_hit_share"),
    ]),
    "holdout3_secondary": ("v14", [
        row("P_T vs C3 (Recall@10)", INSTACART3, H3, "policy:P_T_vs_final_lr_ucf_recall"),
        row("P_T vs C3 (new-item Recall@10)", INSTACART3, H3, "policy:P_T_vs_final_lr_ucf_new_recall"),
        row("P_T vs C3 (repeat Recall@10)", INSTACART3, H3, "policy:P_T_vs_final_lr_ucf_repeat_recall"),
        row("r_short = 8, T = 4 vs C3 (NDCG@10)", INSTACART3, H3, "policy:P_T4_r8_vs_final_lr_ucf_ndcg"),
        row("r_short = 8, T = 4 vs C3 (new-item hit share@10)", INSTACART3, H3, "policy:P_T4_r8_vs_final_lr_ucf_new_hit_share"),
        row("T = 6 vs C3 (NDCG@10)", INSTACART3, H3, "policy:P_T6_r9_vs_final_lr_ucf_ndcg"),
        row("T = 6 vs C3 (new-item hit share@10)", INSTACART3, H3, "policy:P_T6_r9_vs_final_lr_ucf_new_hit_share"),
        row("T = 10 vs C3 (NDCG@10)", INSTACART3, H3, "policy:P_T10_r9_vs_final_lr_ucf_ndcg"),
        row("T = 10 vs C3 (new-item hit share@10)", INSTACART3, H3, "policy:P_T10_r9_vs_final_lr_ucf_new_hit_share"),
    ]),
    "holdout3_replication": ("v13", [
        row("H6 (repeat Recall@10)", INSTACART3, H3, "H6_repeat_recall"),
        row("H7' (NDCG@10)", INSTACART3, H3, "H7prime_ndcg"),
        row("H7' (Recall@10, reported)", INSTACART3, H3, "H7prime_recall"),
        row("H8 (new-item Recall@10)", INSTACART3, H3, "H8_new_recall"),
    ]),
    "holdout3_complete_journey": ("v14", [
        row("H11 (NDCG@10, non-inferiority)", CJ_REFIT, H3, "H11_ndcg", "complete_journey"),
        row("H12 (new-item hit share@10)", CJ_REFIT, H3, "H12_new_hit_share", "complete_journey"),
        row("H13 (new-item hit share@10, S vs flat popularity)", CJ_REFIT, H3, "H13_new_hit_share", "complete_journey"),
        row("H11 (NDCG@10, non-inferiority)", CJ_TRANSFER, H3, "H11_ndcg", "complete_journey", "instacart"),
        row("H12 (new-item hit share@10)", CJ_TRANSFER, H3, "H12_new_hit_share", "complete_journey", "instacart"),
        row("H13 (new-item hit share@10, S vs flat popularity)", CJ_TRANSFER, H3, "H13_new_hit_share", "complete_journey", "instacart"),
    ]),
    "holdout2_primary": ("v13", [
        row("H6 (repeat Recall@10)", INSTACART2, H2, "H6_repeat_recall"),
        row("H7' (NDCG@10)", INSTACART2, H2, "H7prime_ndcg"),
        row("H7' (Recall@10, reported)", INSTACART2, H2, "H7prime_recall"),
        row("H8 (new-item Recall@10)", INSTACART2, H2, "H8_new_recall"),
    ]),
    "holdout2_decomposition": ("v13", [
        row("H7 (NDCG@10): C1 vs C0", INSTACART2, H2, "H7_ndcg"),
        row("H7 (Recall@10): C1 vs C0", INSTACART2, H2, "H7_recall"),
        row("H9 (NDCG@10): C3 vs C1", INSTACART2, H2, "H9_ndcg"),
        row("H9 (Recall@10): C3 vs C1", INSTACART2, H2, "H9_recall"),
        row("C3 vs personal frequency (NDCG@10)", INSTACART2, H2, "integration:final_lr_ucf_vs_personal_frequency_ndcg"),
        row("E2 vs flat item-CF (new-item Recall@10)", INSTACART2, H2, "integration:usercf_vs_itemcf_flat_new_recall"),
    ]),
    "holdout2_complete_journey": ("v13", [
        row("H6 (repeat Recall@10)", CJ_REFIT, H2, "H6_repeat_recall", "complete_journey"),
        row("H7' (NDCG@10)", CJ_REFIT, H2, "H7prime_ndcg", "complete_journey"),
        row("H7' (Recall@10, reported)", CJ_REFIT, H2, "H7prime_recall", "complete_journey"),
        row("H8 (new-item Recall@10)", CJ_REFIT, H2, "H8_new_recall", "complete_journey"),
        row("H6 (repeat Recall@10)", CJ_TRANSFER, H2, "H6_repeat_recall", "complete_journey", "instacart"),
        row("H7' (NDCG@10)", CJ_TRANSFER, H2, "H7prime_ndcg", "complete_journey", "instacart"),
        row("H7' (Recall@10, reported)", CJ_TRANSFER, H2, "H7prime_recall", "complete_journey", "instacart"),
        row("H8 (new-item Recall@10)", CJ_TRANSFER, H2, "H8_new_recall", "complete_journey", "instacart"),
    ]),
    "holdout_v12": ("v12", [
        row("H1 (repeat Recall@10)", "Instacart holdout", H1, "H1_repeat_recall"),
        row("H2 (new-item Recall@10)", "Instacart holdout", H1, "H2_new_recall"),
        row("H3 (NDCG@10)", "Instacart holdout", H1, "H3_ndcg"),
        row("H3 (Recall@10)", "Instacart holdout", H1, "H3_recall"),
        row("H4 (category hit@10)", "Instacart holdout", H1, "H4_category_hit"),
        row("H5 (new-item Recall@10, secondary)", "Instacart holdout", H1, "H5"),
        row("H1 (repeat Recall@10)", CJ_REFIT, H1, "H1_repeat_recall", "complete_journey"),
        row("H2 (new-item Recall@10)", CJ_REFIT, H1, "H2_new_recall", "complete_journey"),
        row("H3 (NDCG@10)", CJ_REFIT, H1, "H3_ndcg", "complete_journey"),
        row("H3 (Recall@10)", CJ_REFIT, H1, "H3_recall", "complete_journey"),
        row("H4 (category hit@10)", CJ_REFIT, H1, "H4_category_hit", "complete_journey"),
        row("H5 (new-item Recall@10, secondary)", CJ_REFIT, H1, "H5", "complete_journey"),
        row("H1 (repeat Recall@10)", CJ_TRANSFER, H1, "H1_repeat_recall", "complete_journey", "instacart"),
        row("H2 (new-item Recall@10)", CJ_TRANSFER, H1, "H2_new_recall", "complete_journey", "instacart"),
        row("H3 (NDCG@10)", CJ_TRANSFER, H1, "H3_ndcg", "complete_journey", "instacart"),
        row("H3 (Recall@10)", CJ_TRANSFER, H1, "H3_recall", "complete_journey", "instacart"),
        row("H4 (category hit@10)", CJ_TRANSFER, H1, "H4_category_hit", "complete_journey", "instacart"),
    ]),
}
# Column headers of every table format.
HEADERS = {
    "v15": ["Hypothesis (metric)", "Dataset", "Users (n)", "Comparator -> model", "diff", "95 % CI", "97.5 % CI",
            "Decision (95 % / 97.5 %)"],
    "v14": ["Hypothesis (metric)", "Dataset", "Users (n)", "Comparator -> model", "diff", "95 % CI", "98.33 % CI",
            "Decision (95 % / 98.33 %)"],
    "v13": ["Hypothesis (metric)", "Dataset", "Users (n)", "Comparator -> model", "diff", "95 % CI", "99.375 % CI",
            "Decision (95 % / 99.375 %)"],
    "v12": ["Hypothesis (metric)", "Dataset", "Users (n)", "Comparator -> model", "diff", "95 % CI", "Decision"],
}


# ---------------------------------------------------------------------------
# rendering from T1
# ---------------------------------------------------------------------------
def fmt_diff(value: float) -> str:
    '''Text of the difference ``value``: signed with four decimals, "0.0000" when it rounds to 0.'''
    return "0.0000" if round(float(value), 4) == 0 else f"{float(value):+.4f}"


def fmt_ci(lo: float, hi: float) -> str:
    '''Text of the interval from ``lo`` to ``hi`` with four decimals in square brackets.'''
    return f"[{float(lo):.4f}, {float(hi):.4f}]"


def t1_row(t1: pd.DataFrame, spec: dict) -> pd.Series:
    """The single current T1 row selected by ``spec`` (stage, id, dataset, transfer_from)."""
    # Select the current rows that match the stage, id, dataset and transfer source of the spec.
    hit = t1[(t1["stage"] == spec["stage"]) & (t1["id"] == spec["id"]) & (t1["dataset"] == spec["dataset"])
             & (t1["transfer_from"].fillna("") == spec["transfer_from"]) & t1["current"].astype(bool)]
    # Fail unless exactly one row matches.
    if len(hit) != 1:
        raise LookupError(f"{len(hit)} current T1 rows for {spec}")
    return hit.iloc[0]


def render_cells(spec: dict, entry: pd.Series, fmt: str) -> list[str]:
    """README cells of one row: label, dataset, n, means, diff, CI(s), decision(s)."""
    # Format the cells that all table formats share, up to the 95 % CI.
    cells = [spec["label"], spec["dataset_text"], f"{int(entry['users_n']):,}",
             f"{float(entry['mean_comparator']):.4f} -> {float(entry['mean_model']):.4f}", fmt_diff(entry["diff"]),
             fmt_ci(entry["ci95_lo"], entry["ci95_hi"])]
    # Read the decision at the 95 % level, empty when T1 has none.
    decision95 = str(entry["decision"]) if isinstance(entry["decision"], str) and entry["decision"] else ""
    # Append the second CI and the decision cell that belong to the table format.
    if fmt == "v12":
        cells.append(decision95 or "(CI only)")
    elif fmt == "v15":
        decision975 = str(entry["decision_975"]) if isinstance(entry["decision_975"], str) and entry["decision_975"] else ""
        cells.append(fmt_ci(entry["ci975_lo"], entry["ci975_hi"]))
        cells.append(f"{decision95} / {decision975}" if decision95 else "(CI only)")
    elif fmt == "v14":
        decision98 = str(entry["decision_9833"]) if isinstance(entry["decision_9833"], str) and entry["decision_9833"] else ""
        cells.append(fmt_ci(entry["ci9833_lo"], entry["ci9833_hi"]))
        cells.append(f"{decision95} / {decision98}" if decision95 else "(CI only)")
    else:
        decision99 = str(entry["decision_99375"]) if isinstance(entry["decision_99375"], str) and entry["decision_99375"] else ""
        cells.append(fmt_ci(entry["ci99375_lo"], entry["ci99375_hi"]))
        cells.append(f"{decision95} / {decision99}" if decision95 else "(CI only)")
    return cells


def render_table(name: str, t1: pd.DataFrame) -> list[str]:
    """Markdown lines of table ``name`` (marker, header, separator, rows)."""
    # Look up the format and the row specifications of the table.
    fmt, rows = TABLES[name]
    header = HEADERS[fmt]
    # Start with the marker line, the header row and the separator row.
    lines = [MARKER.format(name=name), "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    # Add one Markdown row per row specification, filled from its T1 row.
    for spec in rows:
        lines.append("| " + " | ".join(render_cells(spec, t1_row(t1, spec), fmt)) + " |")
    return lines


# ---------------------------------------------------------------------------
# reading and checking the README
# ---------------------------------------------------------------------------
def split_row(line: str) -> list[str]:
    '''Cells of the Markdown table row ``line``, stripped of the outer bars and of spaces.'''
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def readme_table(text: str, name: str) -> list[str] | None:
    """The Markdown table lines (header, separator, rows) that follow the marker of ``name`` in the README ``text``."""
    marker = MARKER.format(name=name)
    lines = text.splitlines()
    # Find the marker line and return None when the README has none.
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration:
        return None
    table = []
    for line in lines[start + 1:]:
        # Skip non-table lines before the table and stop at the first one after it.
        if not line.strip().startswith("|"):
            if table:
                break
            continue
        # Collect the table line.
        table.append(line)
    return table


def check(t1: pd.DataFrame, readme: str) -> int:
    """Compare every table; print each mismatch; return the mismatch count."""
    mismatches = 0
    for name in TABLES:
        # Render the expected table without its marker line and read the table of the README.
        expected = render_table(name, t1)[1:]
        found = readme_table(readme, name)
        # Count every expected data row as a mismatch when the marker is missing.
        if found is None:
            print(f"{name}: marker {MARKER.format(name=name)} not found in README.md")
            mismatches += len(expected) - 2
            continue
        for i in range(max(len(expected), len(found))):
            # Split the expected line and the README line into cells.
            exp = split_row(expected[i]) if i < len(expected) else None
            got = split_row(found[i]) if i < len(found) else None
            # Count a line that exists on one side only as one mismatch.
            if exp is None or got is None:
                print(f"{name} line {i}: {'missing in README' if got is None else 'extra in README'}: {exp or got}")
                mismatches += 1
                continue
            if i == 1:  # separator row: alignment markers are free
                continue
            # Compare the two lines cell by cell and count every differing cell.
            for j in range(max(len(exp), len(got))):
                e = exp[j] if j < len(exp) else None
                g = got[j] if j < len(got) else None
                if e != g:
                    print(f"{name} line {i} cell {j}: README {g!r} != T1 {e!r}")
                    mismatches += 1
        # Print the number of data rows of the table.
        print(f"{name}: {len(expected) - 2} rows compared")
    return mismatches


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    '''
    Parse ``argv`` (default: the command line), read T1 and either print the rendered
    tables (--render) or check the README and exit with code 1 on a mismatch.
    '''
    # Parse the command-line options.
    parser = argparse.ArgumentParser(description="README result tables vs T1")
    parser.add_argument("--render", action="store_true", help="print the tables instead of checking the README")
    parser.add_argument("--t1", default=str(T1_PATH))
    parser.add_argument("--readme", default=str(README_PATH))
    args = parser.parse_args(argv)
    # Read T1 with empty cells kept as empty strings.
    t1 = pd.read_csv(args.t1, keep_default_na=False, na_values=[])
    # Convert the current column to booleans.
    t1["current"] = t1["current"].astype(str).str.lower().eq("true")
    # Print every rendered table and stop when --render is given.
    if args.render:
        for name in TABLES:
            print("\n".join(render_table(name, t1)) + "\n")
        return
    # Compare the README with T1 and print the mismatch count.
    mismatches = check(t1, Path(args.readme).read_text(encoding="utf-8"))
    print(f"README vs T1: {mismatches} mismatches")
    # Exit with code 1 when any cell differs.
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
