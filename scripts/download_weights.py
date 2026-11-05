#!/usr/bin/env python3
"""Restore the pinned public model release, verifying every file with SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).with_name('model_release.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'])


def restore(destination, check=False):
    manifest = json.loads(RELEASE.read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('The model release must pin an immutable Hub commit.')
    checked = []
    for record in manifest['files']:
        relative = PurePosixPath(record['path'])
        if (relative.is_absolute() or '..' in relative.parts or '\\' in record['path']
                or len(relative.parts) != 2
                or relative.parts[0] not in {'rmnet', 'rmnet_bert_lora'}
                or relative.as_posix() != record['path']
                or record['bytes'] < 0
                or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Invalid model release manifest entry.')
        target = destination / relative
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError(f'Symlink in destination path: {target}')
        if target.exists():
            if not matches(target, record):
                raise ValueError(f'Existing file differs from the release; not overwritten: {target}')
        elif check:
            raise FileNotFoundError(f'Missing model asset: {target}')
        else:
            checked.append((target, record))
    for target, record in checked:
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(manifest['repository'], record['path'],
                                    revision=manifest['revision'], token=False))
        if not matches(cached, record):
            raise ValueError(f'Download checksum mismatch: {record["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='.emomosaic-', dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as output, cached.open('rb') as stream:
                shutil.copyfileobj(stream, output)
            if not matches(temporary, record):
                raise ValueError(f'Copied file checksum mismatch: {record["path"]}')
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Restored {record["path"]}', flush=True)
    print(f'Verified {len(manifest["files"])} model assets at {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'code/models')
    parser.add_argument('--check', action='store_true', help='Verify local files without network access')
    args = parser.parse_args()
    restore(args.destination.absolute(), args.check)


if __name__ == '__main__':
    main()
