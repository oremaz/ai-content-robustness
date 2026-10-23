import json
import zipfile

import pytest

from scripts.extract_artifact import extract


def archive(directory):
    with zipfile.ZipFile(directory / "artifact-dataset.zip", "w") as handle:
        handle.writestr("real/one.png", b"one complete image")
        handle.writestr("fake/two.jpg", b"another complete image")
        handle.writestr("README.txt", "dataset metadata")


def test_extraction_repairs_partial_files_and_requires_all_archive_entries(tmp_path):
    archive(tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "real/one.png").write_bytes(b"bad incomplete image")
    result = extract(tmp_path, workers=2)
    assert result["validated_files"] == 3
    assert result["image_count"] == 2
    assert result["crc_verified"] is True
    assert (tmp_path / "real/one.png").read_bytes() == b"one complete image"
    assert json.loads((tmp_path / ".artifact-complete.json").read_text()) == result
    assert extract(tmp_path, workers=2) == result


def test_extraction_rejects_unmanifested_images_before_readiness(tmp_path):
    archive(tmp_path)
    (tmp_path / "unexpected.jpg").write_bytes(b"extra data")
    with pytest.raises(ValueError, match="absent from archive manifest"):
        extract(tmp_path, workers=2)
    assert not (tmp_path / ".artifact-complete.json").exists()


def test_invalid_archive_never_creates_readiness_marker(tmp_path):
    (tmp_path / "artifact-dataset.zip").write_bytes(b"partial archive")
    with pytest.raises(zipfile.BadZipFile):
        extract(tmp_path, workers=1)
    assert not (tmp_path / ".artifact-complete.json").exists()
