#!/usr/bin/env python3
"""Restore the pinned public EmoMosaic release and verify every source file.

Only missing files are downloaded. Existing mismatches are reported, never
overwritten. --check reads local files and the small pinned manifest only;
--manifest makes that check fully offline. No Hugging Face token is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "config-h/EmoMosaic"
REVISION = "c9c27b40a226f9f14ae05dde5b8ea4d5ae65ae87"
MANIFEST_SHA256 = "418e34d051a6a8a33d832512743ee87790e0295fe1adc2616f4c38e7ec6b61c9"
FILE_COUNT = 244
TOTAL_BYTES = 4_077_821_811
MANIFEST_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/manifest.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_manifest(path: Path | None) -> tuple[bytes, list[dict]]:
    if path is None:
        request = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "EmoMosaic-data-restore/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(1_048_577)
    else:
        content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
        raise ValueError("The manifest SHA-256 differs from the pinned release; no data were changed.")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not {"file_path", "size_bytes", "sha256"}.issubset(reader.fieldnames or []):
        raise ValueError("The release manifest does not contain the expected columns.")
    rows = []
    seen = set()
    for row in reader:
        remote = PurePosixPath(row["file_path"])
        if (remote.is_absolute() or ".." in remote.parts or "\\" in row["file_path"]
                or len(remote.parts) < 2 or remote.parts[0] != "data"
                or remote.as_posix() != row["file_path"]):
            raise ValueError("Unsafe data path in the release manifest.")
        relative = remote.relative_to("data").as_posix()
        if relative in seen:
            raise ValueError(f"Duplicate release path: {relative}")
        size = int(row["size_bytes"])
        if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid size or hash for {relative}")
        rows.append({"remote": remote.as_posix(), "relative": relative,
                     "bytes": size, "sha256": row["sha256"]})
        seen.add(relative)
    if len(rows) != FILE_COUNT or sum(row["bytes"] for row in rows) != TOTAL_BYTES:
        raise ValueError("Manifest file count or total size differs from the pinned release.")
    return content, rows


def target_path(directory: Path, row: dict) -> Path:
    path = directory / row["relative"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Destination escapes the data directory: {row['relative']}")
    # A restored release uses ordinary files and directories, never symlink targets.
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Symlink in destination path: {row['relative']}")
        if item == directory:
            break
    return path


def file_error(path: Path, row: dict) -> str | None:
    if not path.is_file():
        return "not a regular file"
    if path.stat().st_size != row["bytes"]:
        return "size mismatch"
    if digest(path) != row["sha256"]:
        return "SHA-256 mismatch"
    return None


def install_file(source: Path, destination: Path, row: dict) -> None:
    error = file_error(source, row)
    if error:
        raise ValueError(f"Downloaded {row['relative']}: {error}; not installed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".emomosaic-", suffix=".part", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
        error = file_error(temporary, row)
        if error:
            raise ValueError(f"Copied {row['relative']}: {error}; not installed.")
        try:
            # Link within one directory publishes the complete file without replacing
            # a destination created by another process while the download ran.
            os.link(temporary, destination)
        except FileExistsError:
            error = file_error(destination, row)
            if error:
                raise ValueError(f"Destination appeared during download: {row['relative']} ({error}); preserved.")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all 244 local data files; download no data")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/source_data",
                        help="Restore destination (default: project data/source_data)")
    parser.add_argument("--manifest", type=Path,
                        help="Local copy of the pinned manifest.csv; enables a fully offline --check")
    args = parser.parse_args()
    directory = args.data_dir.absolute()
    saved_manifest = directory.parent / ".emomosaic" / f"manifest-{REVISION}.csv"
    manifest_path = args.manifest or (saved_manifest if saved_manifest.is_file() else None)
    try:
        manifest_bytes, rows = read_manifest(manifest_path)
        missing = []
        problems = []
        for row in rows:
            path = target_path(directory, row)
            if not path.exists():
                missing.append(row)
            elif error := file_error(path, row):
                problems.append(f"{row['relative']}: {error}")
        if problems:
            for problem in problems[:20]:
                print(problem, file=sys.stderr)
            raise ValueError(f"{len(problems)} existing file(s) differ from the release. No data were changed; move them aside explicitly before retrying.")
        if args.check:
            if missing:
                for row in missing[:20]:
                    print(f"Missing: {row['relative']}", file=sys.stderr)
                raise ValueError(f"{len(missing)} of {FILE_COUNT} source files are missing. Run this script without --check to restore them.")
            print(f"PASS: {FILE_COUNT} files, {TOTAL_BYTES:,} bytes; every SHA-256 matches {REVISION}.")
            return 0
        if missing:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ValueError("Install huggingface_hub in this interpreter: python -m pip install huggingface_hub") from None
            print(f"Restoring {len(missing)} missing files from {REPOSITORY}@{REVISION}.", flush=True)
            for index, row in enumerate(missing, 1):
                print(f"[{index}/{len(missing)}] {row['relative']} ({row['bytes']:,} bytes)", flush=True)
                cached = hf_hub_download(repo_id=REPOSITORY, repo_type="dataset", revision=REVISION,
                                         filename=row["remote"], token=False)
                install_file(Path(cached), target_path(directory, row), row)
        saved_manifest.parent.mkdir(parents=True, exist_ok=True)
        saved_manifest.write_bytes(manifest_bytes)
        print(f"PASS: {FILE_COUNT} files, {TOTAL_BYTES:,} bytes verified; restored {len(missing)} files to {directory}.")
        print(f"Offline verification: python scripts/download_data.py --check --manifest {saved_manifest}")
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
