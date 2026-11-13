# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Tests for scripts/make_holdout.py: argument parsing, output paths and the exclusion
          set (read vs reconstructed).
Inputs  : The script scripts/make_holdout.py loaded as a module, an eligible set of 200 user
          ids, and profile parquet files and selection reports written to the pytest tmp_path
          folder.
Outputs : None; the tests assert on the parsed arguments, the report and output paths, the
          exclusion sets and the id digest of the script.
Usage   :
    python -m pytest tests/test_make_holdout.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import importlib.util
import json

import pandas as pd
import pytest

from nbr.config import REPO_ROOT


# ---------------------------------------------------------------------------
# script loading and helpers
# ---------------------------------------------------------------------------
def _load_script():
    '''Load scripts/make_holdout.py as a module and return it.'''
    # Build the import spec from the path of the script.
    path = REPO_ROOT / "scripts" / "make_holdout.py"
    spec = importlib.util.spec_from_file_location("make_holdout", path)
    # Create the module object and execute the script in it.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mh = _load_script()


def _write_profile(data_dir, profile, users):
    '''Write an eval_target.parquet with the ids in ``users`` for ``profile`` under ``data_dir``.'''
    # Create the Instacart folder of the profile.
    base = data_dir / profile / "instacart"
    base.mkdir(parents=True)
    # Write one row per user in sorted id order.
    frame = pd.DataFrame({"user_id": sorted(users), "basket_id": ["b"] * len(users), "item_id": ["i"] * len(users)})
    frame.to_parquet(base / "eval_target.parquet", index=False)


def _write_report(report_dir, profile, drawn, seed, size, **extra):
    '''Write the selection report of ``profile`` to ``report_dir``.

    The report holds ``seed``, ``size``, the count and digest of the ids in ``drawn`` and the
    keys of ``extra``.
    '''
    # Create the report folder when it does not exist yet.
    report_dir.mkdir(parents=True, exist_ok=True)
    # Collect the report fields and add the extra keys.
    report = {"seed": seed, "size": size, "selected_users": len(drawn), "selected_users_sha256": mh.ids_digest(drawn)}
    report.update(extra)
    # Write the report as JSON to the path the script uses for this profile.
    mh.report_path(profile, report_dir).write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# argument parsing and paths
# ---------------------------------------------------------------------------
def test_parse_args_defaults_are_the_v12_holdout_draw():
    # Parse only the required option.
    args = mh.parse_args(["--raw-dir", "raw"])
    # Expect the default name, exclusions, size and seed and no expected remaining count.
    assert args.raw_dir == "raw"
    assert args.name == "holdout"
    assert args.exclude == ["medium", "sample"]
    assert (args.size, args.seed, args.expect_remaining) == (20000, 42, None)


def test_parse_args_holdout2():
    # Parse every option, with blanks and a repeated profile in the exclusion list.
    args = mh.parse_args(["--raw-dir", "raw", "--name", "holdout2", "--exclude", " sample, medium,holdout,sample, ",
                          "--size", "100", "--seed", "7", "--expect-remaining", "89936"])
    # Expect the given name and numbers and the cleaned exclusion list.
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
    # Expect SystemExit for each argument list of the parameter table.
    with pytest.raises(SystemExit):
        mh.parse_args(argv)


def test_parse_exclude():
    # Expect the profile names in the given order without blanks and repeats.
    assert mh.parse_exclude("medium,sample") == ["medium", "sample"]
    assert mh.parse_exclude("sample,medium,holdout") == ["sample", "medium", "holdout"]
    assert mh.parse_exclude(",medium,,medium , sample") == ["medium", "sample"]
    assert mh.parse_exclude("") == []


def test_report_path_keeps_v12_name(tmp_path):
    # Expect <name>_selection.json in reports/holdout_prep or in the given report folder.
    assert mh.report_path("holdout") == REPO_ROOT / "reports" / "holdout_prep" / "holdout_selection.json"
    assert mh.report_path("holdout2") == REPO_ROOT / "reports" / "holdout_prep" / "holdout2_selection.json"
    assert mh.report_path("holdout2", tmp_path) == tmp_path / "holdout2_selection.json"


def test_output_dirs_follow_data_dir_and_name(tmp_path):
    # Ask for the output folders of a profile under a temporary data folder.
    processed, artifacts = mh.output_dirs("holdout2", tmp_path)
    # Expect the processed folder under the data folder and the artifacts under the repository.
    assert processed == tmp_path / "holdout2"
    assert artifacts == REPO_ROOT / "artifacts" / "holdout2"
    assert mh.output_dirs("holdout", REPO_ROOT / "data" / "processed")[0] == REPO_ROOT / "data" / "processed" / "holdout"


def test_check_remaining():
    # Expect no error without an expected count or with an equal count.
    mh.check_remaining(89936, None)
    mh.check_remaining(89936, 89936)
    # Expect SystemExit when the remaining count differs from the expected one.
    with pytest.raises(SystemExit):
        mh.check_remaining(89935, 89936)


# ---------------------------------------------------------------------------
# exclusion set
# ---------------------------------------------------------------------------
@pytest.fixture
def eligible():
    '''Return the eligible user ids "1" to "200" as a set of strings.'''
    return {str(i) for i in range(1, 201)}


def test_exclusion_reads_profiles_without_report(tmp_path, eligible):
    # Write two overlapping profiles to disk and no selection report.
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    # Expect each set to be the users read from the profile and the union of both.
    sets = mh.exclusion_sets(["medium", "sample"], eligible, data_dir, report_dir)
    assert sets == {"medium": medium, "sample": sample}
    assert mh.exclusion_users(["medium", "sample"], eligible, data_dir, report_dir) == medium | sample


def test_exclusion_reconstructs_profile_with_report_without_reading_it(tmp_path, eligible):
    # Write two profiles to disk and only a selection report for a third draw of 50 users.
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    holdout = mh.ORIGINAL_SAMPLE_IDS(eligible - medium - sample, 50, 42)
    _write_report(report_dir, "holdout", holdout, seed=42, size=50)  # v1.2 report: no excluded_profiles key
    assert not (data_dir / "holdout").exists()  # nothing to read: the draw is re-run instead

    # Expect the third set to equal the drawn users and the union to hold all three sets.
    sets = mh.exclusion_sets(["sample", "medium", "holdout"], eligible, data_dir, report_dir)
    assert sets["holdout"] == holdout
    assert len(holdout) == 50 and not holdout & (medium | sample)
    union = mh.exclusion_users(["sample", "medium", "holdout"], eligible, data_dir, report_dir)
    assert union == medium | sample | holdout
    assert len(eligible - union) == 200 - len(union)


def test_exclusion_reconstruction_uses_recorded_exclusions(tmp_path, eligible):
    # Write one profile and a report whose excluded_profiles key names that profile.
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    _write_profile(data_dir, "medium", medium)
    other = mh.ORIGINAL_SAMPLE_IDS(eligible - medium, 30, 3)
    _write_report(report_dir, "other", other, seed=3, size=30, excluded_profiles=["medium"])
    # Expect the reconstructed set to equal the users drawn after that exclusion.
    assert mh.exclusion_sets(["other"], eligible, data_dir, report_dir)["other"] == other


def test_exclusion_digest_mismatch_stops_without_fallback(tmp_path, eligible):
    # Write three profiles to disk and a report for the third one.
    data_dir, report_dir = tmp_path / "data", tmp_path / "reports"
    medium = {str(i) for i in range(1, 21)}
    sample = {str(i) for i in range(15, 31)}
    _write_profile(data_dir, "medium", medium)
    _write_profile(data_dir, "sample", sample)
    _write_profile(data_dir, "holdout", {"150", "151"})  # present on disk but must not be used
    drawn = mh.ORIGINAL_SAMPLE_IDS(eligible - medium - sample, 50, 42)
    _write_report(report_dir, "holdout", drawn, seed=42, size=50)
    # Replace the digest in the report by 64 zeros.
    report = mh.report_path("holdout", report_dir)
    document = json.loads(report.read_text(encoding="utf-8"))
    document["selected_users_sha256"] = "0" * 64
    report.write_text(json.dumps(document), encoding="utf-8")
    # Expect RuntimeError for the wrong digest.
    with pytest.raises(RuntimeError, match="does not match"):
        mh.exclusion_sets(["medium", "sample", "holdout"], eligible, data_dir, report_dir)
    # Restore the digest, set a wrong user count and expect RuntimeError again.
    document["selected_users_sha256"] = mh.ids_digest(drawn)
    document["selected_users"] = 49
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        mh.exclusion_sets(["holdout"], eligible, data_dir, report_dir)


# ---------------------------------------------------------------------------
# id digest
# ---------------------------------------------------------------------------
def test_ids_digest_is_order_independent():
    # Expect the same digest for the same ids in any order and another one for other ids.
    assert mh.ids_digest({"b", "a"}) == mh.ids_digest({"a", "b"})
    assert mh.ids_digest({"a"}) != mh.ids_digest({"b"})
