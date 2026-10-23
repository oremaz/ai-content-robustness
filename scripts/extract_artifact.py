"""Validate and resume full ArtiFact extraction without accepting partial data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import time
import zipfile
import zlib


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_crc(path):
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(directory, workers=16):
    directory = Path(directory).resolve()
    archive_path = directory / "artifact-dataset.zip"
    marker = directory / ".artifact-complete.json"
    archive_sha256 = file_digest(archive_path)
    marker.unlink(missing_ok=True)
    progress_path = directory / ".extract-progress.json"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = set()
        expected_images = 0
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
            if entry.filename in {archive_path.name, marker.name, progress_path.name}:
                raise ValueError(f"Archive entry conflicts with extraction metadata: {entry.filename}")
            if entry.is_dir():
                continue
            if entry.filename in names:
                raise ValueError(f"Duplicate archive entry: {entry.filename}")
            names.add(entry.filename)
            expected_images += path.suffix.lower() in IMAGE_EXTENSIONS
        expected_files = len(names)

        def process(entry):
            target = directory / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == entry.file_size and file_crc(target) == entry.CRC:
                return False
            temporary = target.with_name(f".{target.name}.extract-{os.getpid()}-{threading.get_ident()}.part")
            try:
                with archive.open(entry) as source, temporary.open("wb") as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                if temporary.stat().st_size != entry.file_size:
                    raise ValueError(f"Incomplete extracted file: {entry.filename}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            return True

        completed = written = 0
        last_progress = time.monotonic()
        pending = set()

        def collect(futures):
            nonlocal completed, written, last_progress
            for future in futures:
                written += bool(future.result())
                completed += 1
            if time.monotonic() - last_progress >= 30 or completed == expected_files:
                progress = {"archive_sha256": archive_sha256, "expected_files": expected_files,
                            "validated_files": completed, "written_files": written}
                write_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for entry in entries:
                if entry.is_dir():
                    continue
                pending.add(executor.submit(process, entry))
                if len(pending) >= workers * 4:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(finished)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)

        actual_images = 0
        for root, directories, filenames in os.walk(directory):
            for filename in filenames:
                path = Path(root) / filename
                if filename.startswith(".") and ".extract-" in filename and filename.endswith(".part"):
                    path.unlink()
                elif path.suffix.lower() in IMAGE_EXTENSIONS:
                    relative = path.relative_to(directory).as_posix()
                    if relative not in names:
                        raise ValueError(f"Image absent from archive manifest: {relative}")
                    actual_images += 1
        if completed != expected_files or actual_images != expected_images:
            raise ValueError(f"Extraction count mismatch: files {completed}/{expected_files}; images {actual_images}/{expected_images}")
        result = {"complete": True, "archive_sha256": archive_sha256,
                  "archive_entries": len(entries), "validated_files": completed,
                  "image_count": actual_images, "crc_verified": True}
        write_json(marker, result)
        print(json.dumps(result), flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    extract(arguments.directory, arguments.workers)
