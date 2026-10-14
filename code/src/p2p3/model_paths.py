"""Resolve renamed model assets while preserving historical hash records."""
import hashlib
import json
from pathlib import Path

from project_paths import ROOT, resolve_asset
OLD_DIR = 'models/rmnet_improved'
MODEL_DIR = 'models/rmnet_bert_lora'


def current_model_paths(value):
    """Translate paths in an in-memory copy; never rewrite frozen records."""
    if isinstance(value, dict):
        return {current_model_paths(k): current_model_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [current_model_paths(v) for v in value]
    if isinstance(value, str) and (value == OLD_DIR or value.startswith(OLD_DIR + '/')):
        return MODEL_DIR + value[len(OLD_DIR):]
    return value


def resolve_model_path(path, root=ROOT):
    """Support both current workspace paths and self-contained older bundles."""
    root = Path(root)
    path = resolve_asset(path) if root.resolve() == ROOT.resolve() else Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        source_name = path.relative_to(root).as_posix()
    except ValueError:
        source_name = ''
    if source_name.startswith('src/P2P3_model/'):
        path = root / source_name.replace('src/P2P3_model/', 'src/p2p3/', 1)
    if path.exists():
        return path
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        return path
    for source, target in ((OLD_DIR, MODEL_DIR), (MODEL_DIR, OLD_DIR)):
        if name == source or name.startswith(source + '/'):
            candidate = root / (target + name[len(source):])
            if candidate.exists():
                return candidate
    return path


def asset_matches(path, expected_sha256, root=ROOT):
    """Accept exact bytes or an explicitly recorded rename migration only."""
    root = Path(root)
    path = resolve_model_path(path, root)
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual == expected_sha256:
        return True
    migration = root / MODEL_DIR / 'path_migration.json'
    records = json.loads(migration.read_text())['files'] if migration.is_file() else []
    if any(resolve_model_path(record['path'], root) == path
               and record['before_sha256'] == expected_sha256
               and record['after_sha256'] == actual
               for record in records):
        return True
    # Source-only standalone migration: the preserved parent snapshot must hash
    # to the recorded before value, and any earlier transition must already be
    # evidenced by the original migration manifest. Neither manifest is rewritten.
    source_migration = root / 'outputs/provenance/standalone_source_migration.json'
    if not source_migration.is_file():
        return False
    for record in json.loads(source_migration.read_text())['files']:
        if not record['path'].startswith('src/p2p3/'):
            continue
        if root / record['path'] != path or record['after_sha256'] != actual:
            continue
        snapshot = root / record['before_snapshot']
        if (not snapshot.resolve().is_relative_to((root / 'outputs/provenance').resolve())
                or not snapshot.is_file()
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != record['before_sha256']):
            continue
        if expected_sha256 == record['before_sha256']:
            return True
        if any(resolve_model_path(prior['path'], root) == path
               and prior['before_sha256'] == expected_sha256
               and prior['after_sha256'] == record['before_sha256'] for prior in records):
            return True
    return False
