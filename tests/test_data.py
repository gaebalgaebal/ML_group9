import zipfile

import pytest

from ml_group9.data import discover_archive


def test_archive_is_discovered_by_members_not_filename(tmp_path):
    path = tmp_path / "arbitrary-name.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("first.csv", "x\n1\n")
        archive.writestr("second.csv", "y\n2\n")
    assert discover_archive(tmp_path, {"first.csv", "second.csv"}) == path


def test_archive_discovery_rejects_missing_members(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_archive(tmp_path, {"missing.csv"})
