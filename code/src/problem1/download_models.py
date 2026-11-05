"""Download only frozen, public feature extractors; record immutable revisions."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'models/pretrained'
MODELS = {
    'bert': ('google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594',
             ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']),
    'ctc': ('facebook/wav2vec2-base-960h', '22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
            ['config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json']),
}

def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_models():
    """Validate bundled weights offline against the recorded extraction assets."""
    expected = json.loads(Path(__file__).with_name('model_manifest.json').read_text())
    failures = []
    for item in expected:
        path = BASE / item['path']
        if not path.is_file():
            failures.append(f'Missing: {item["path"]}')
        elif path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            failures.append(f'Size or SHA256 mismatch: {item["path"]}')
    if failures:
        raise SystemExit('\n'.join(failures))
    return {'status': 'passed', 'files_checked': len(expected),
            'bytes_checked': sum(item['bytes'] for item in expected)}


def fetch(item):
    import requests
    name, url = item
    dest = BASE / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    if not dest.exists():
        failures = 0
        while not dest.exists():
            try:
                offset = part.stat().st_size if part.exists() else 0
                with requests.get(url, stream=True, headers={'Range': f'bytes={offset}-{offset+4*1024*1024-1}'}, timeout=(30, 60)) as r:
                    r.raise_for_status()
                    total = int(r.headers['Content-Range'].split('/')[-1]) if r.status_code == 206 else int(r.headers['Content-Length'])
                    if r.status_code == 206:
                        assert int(r.headers['Content-Range'].split()[1].split('-')[0]) == offset
                    with part.open('ab' if r.status_code == 206 else 'wb') as f:
                        for chunk in r.iter_content(1024 * 1024): f.write(chunk)
                if part.stat().st_size == total: part.replace(dest)
                elif part.stat().st_size > total: raise ValueError('Oversized download')
                failures = 0
            except requests.RequestException as exc:
                failures += 1
                print(name, 'retry', failures, type(exc).__name__, flush=True)
                if failures == 12: raise
                time.sleep(2)
    checksum = digest(dest)
    print(name, dest.stat().st_size, flush=True)
    return {'path': name, 'url': url, 'sha256': checksum, 'bytes': dest.stat().st_size}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Check existing models and hashes offline; download nothing')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_models(), indent=2))
        raise SystemExit(0)
    items = [(f'{name}/{f}', f'https://huggingface.co/{repo}/resolve/{rev}/{f}')
             for name, (repo, rev, files) in MODELS.items() for f in files]
    items.append(('face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(fetch, items))
    check_models()
    (BASE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
