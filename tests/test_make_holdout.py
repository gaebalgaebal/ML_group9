"""Tests for scripts/make_holdout.py: argument parsing, output paths and the exclusion set (read vs reconstructed)."""

import importlib.util
import json

import pandas as pd
import pytest

from nbr.config import REPO_ROOT


def _load_script():
    path = REPO_ROOT / "scripts" / "make_holdout.py"
    spec = importlib.util.spec_from_file_location("make_holdout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mh = _load_script()


def _write_profile(data_dir, profile, users):
    base = data_dir / profile / "instacart"
    base.mkdir(parents=True)
    frame = pd.DataFrame({"user_id": sorted(users), "basket_id": ["b"] * len(users), "item_id": ["i"] * len(users)})
    frame.to_parquet(base / "eval_target.parquet", index=False)


def _write_report(report_dir, profile, drawn, seed, size, **extra):
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed, "size": size, "selected_users": len(drawn), "selected_users_sha256": mh.ids_digest(drawn)}
    report.update(extra)
    mh.report_path(profile, report_dir).write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# argument parsing and paths
# ---------------------------------------------------------------------------
def test_parse_args_defaults_are_the_v12_holdout_draw():
    args = mh.parse_args(["--raw-dir", "raw"])
    assert args.raw_dir == "raw"
    assert args.name == "holdout"
    assert args.exclude == ["medium", "sample"]
    assert (args.size, args.seed, args.expect_remaining) == (20000, 42, None)


def test_parse_args_holdout2():
    args = mh.parse_args(["--raw-dir", "raw", "--name", "holdout2", "--exclude", " sample, medium,holdout,sample, ",
                          "--size", "100", "--seed", "7", "--expect-remaining", "89936"])
    assert args.name == "holdout2"
    assert args.exclude == ["sample", "medium", "holdout"]  # order kept, blanks and repeats dropped
    assert (args.size, args.seed, args.expect_remaining) == (100, 7, 89936)


@pytest.mark.parametrize("argv", [
    ["--raw-dir", "raw", "--name", "../x"],
    ["--raw-dir", "raw", "--name", ""],
    ["--raw-dir", "raw", "--exclude", " , "],
    [],
])
def test_parse_args_rejects_bad_name_or_empty_exclusions(argv):
    with pytest.raises(SystemExit):
        mh.parse_args(argv)


def test_parse_exclude():
    assert mh.parse_exclude("medium,sample") == ["medium", "sample"]
    assert mh.parse_exclude("sample,medium,holdout") == ["sample", "medium", "holdout"]
    assert mh.parse_exclude(",medium,,medium , sample") == ["medium", "sample"]
    assert mh.parse_exclude("") == []


def test_report_path_keeps_v12_name(tmp_path):
    assert mh.report_path("holdout") == REPO_ROOT / "reports" / "holdout_prep" / "holdout_selection.json"
    assert mh.report_path("holdout2") == REPO_ROOT / "reports" / "holdout_prep" / "holdout2_selection.json"
    assert mh.report_path("holdout2", tmp_path) == tmp_path / "holdout2_selection.json"


def test_output_dirs_follow_data_dir_and_name(tmp_path):
    processed, artifacts = mh.output_dirs("holdout2", tmp_path)
    assert processed == tmp_path / "holdout2"
    assert artifacts == REPO_ROOT / "artifacts" / "holdout2"
    assert mh.output_dirs("holdout", REPO_ROOT / "data" / "processed")[0] == REPO_ROOT / "data" / "processed" / "holdout"


def test_check_remaining():
    mh.check_remaining(89936, None)
    mh.check_remaining(89936, 89936)
    with pytest.raises(SystemExit):
        mh.check_remaining(89935, 89936)


# ---------------------------------------------------------------------------
# exclusion set
# ---------------------------------------------------------------------------
@pytest.fixture
def eligible():
    return {str(i) for i in range(1, 201)}


def test_exclusion_reads_profiles_without_report(tmp_path, eligible):
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    sets = mh.exclusion_sets(["medium", "sample"], eligible, data_dir, report_dir)
    assert sets == {"medium": medium, "sample": sample}
    assert mh.exclusion_users(["medium", "sample"], eligible, data_dir, report_dir) == medium | sample


def test_exclusion_reconstructs_profile_with_report_without_reading_it(tmp_path, eligible):
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    holdout = mh.ORIGINAL_SAMPLE_IDS(eligible - medium - sample, 50, 42)
    _write_report(report_dir, "holdout", holdout, seed=42, size=50)  # v1.2 report: no excluded_profiles key
    assert not (data_dir / "holdout").exists()  # nothing to read: the draw is re-run instead

    sets = mh.exclusion_sets(["sample", "medium", "holdout"], eligible, data_dir, report_dir)
    assert sets["holdout"] == holdout
    assert len(holdout) == 50 and not holdout & (medium | sample)
    union = mh.exclusion_users(["sample", "medium", "holdout"], eligible, data_dir, report_dir)
    assert union == medium | sample | holdout
    assert len(eligible - union) == 200 - len(union)


def test_exclusion_reconstruction_uses_recorded_exclusions(tmp_path, eligible):
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    _write_profile(data_dir, "medium", medium)
    other = mh.ORIGINAL_SAMPLE_IDS(eligible - medium, 30, 3)
    _write_report(report_dir, "other", other, seed=3, size=30, excluded_profiles=["medium"])
    assert mh.exclusion_sets(["other"], eligible, data_dir, report_dir)["other"] == other


def test_exclusion_digest_mismatch_stops_without_fallback(tmp_path, eligible):
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    _write_profile(data_dir, "holdout", {"150", "151"})  # present on disk but must not be used
    drawn = mh.ORIGINAL_SAMPLE_IDS(eligible - medium - sample, 50, 42)
    _write_report(report_dir, "holdout", drawn, seed=42, size=50)
    report = mh.report_path("holdout", report_dir)
    document = json.loads(report.read_text(encoding="utf-8"))
    document["selected_users_sha256"] = "0" * 64
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        mh.exclusion_sets(["medium", "sample", "holdout"], eligible, data_dir, report_dir)
    document["selected_users_sha256"] = mh.ids_digest(drawn)
    document["selected_users"] = 49
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        mh.exclusion_sets(["holdout"], eligible, data_dir, report_dir)


def test_ids_digest_is_order_independent():
    assert mh.ids_digest({"b", "a"}) == mh.ids_digest({"a", "b"})
    assert mh.ids_digest({"a"}) != mh.ids_digest({"b"})
