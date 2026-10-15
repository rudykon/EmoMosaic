"""Restore same-origin browser models from the pinned EmoMosaic release APK."""
import argparse
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INFERENCE = PROJECT / 'website/assets/inference'


def matches(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == metadata['sha256']


def download_apk(metadata):
    cache = PROJECT / 'validation/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / metadata['filename']
    if matches(target, metadata):
        return target
    request = urllib.request.Request(metadata['url'], headers={'User-Agent': 'EmoMosaic-Pages'})
    for attempt in range(3):
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix='.part', delete=False) as stream:
                partial = Path(stream.name)
                with urllib.request.urlopen(request, timeout=60) as response:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            if not matches(partial, metadata):
                raise ValueError('Downloaded APK does not match its pinned size and SHA-256')
            partial.replace(target)
            return target
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)


def prepare(apk=None):
    release = json.loads((PROJECT / 'website/data/app-release.json').read_text())
    manifest = json.loads((INFERENCE / 'manifest.json').read_text())
    models = {name: meta for name, meta in manifest['assets'].items() if name.startswith('models/')}
    if len(models) != 33:
        raise ValueError('Expected 33 pinned browser model assets')
    for name in models:
        target = (INFERENCE / name).resolve()
        if not target.is_relative_to(INFERENCE / 'models'):
            raise ValueError(f'Model path escapes its directory: {name}')
    if apk is None and all(matches(INFERENCE / name, meta) for name, meta in models.items()):
        print('PASS: all 33 browser model assets already match the manifest')
        return
    apk = apk or download_apk(release['apk'])
    if not matches(apk, release['apk']):
        raise ValueError('APK does not match its pinned size and SHA-256')
    # Read only manifest-listed members; never extract arbitrary APK paths.
    with zipfile.ZipFile(apk) as archive:
        for name, meta in models.items():
            target = INFERENCE / name
            if matches(target, meta):
                continue
            member = 'assets/' + name
            if archive.getinfo(member).file_size != meta['bytes']:
                raise ValueError(f'Model size mismatch: {name}')
            content = archive.read(member)
            if hashlib.sha256(content).hexdigest() != meta['sha256']:
                raise ValueError(f'Model SHA-256 mismatch: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print('PASS: restored 33 browser model assets from the verified release APK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk', type=Path, help='Use an already downloaded, checksum-matching APK')
    prepare(parser.parse_args().apk)
