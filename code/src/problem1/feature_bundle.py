"""Lossless full P1 feature storage; no feature extraction or model execution.

Byte shuffling groups equal-significance bytes before XZ compression. The
inverse restores each original NPY member, then the original NPZ file. SHA256
checks cover both levels. All 300 L0 and 100 L1 files are included, untruncated.
"""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'outputs/problem1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def header(data):
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        _, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        _, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported NPY version: {version}')
    if dtype.hasobject:
        raise ValueError('Object arrays are not allowed')
    return stream.tell(), dtype.itemsize


def restore(bundle, destination):
    meta = json.loads((bundle / 'full_features_manifest.json').read_text())
    compressed = (bundle / 'full_features.bin.xz').read_bytes()
    assert sha(compressed) == meta['payload_sha256']
    data = lzma.decompress(compressed)
    assert len(data) == meta['uncompressed_bytes']
    for record in meta['files']:
        relative = Path(record['file'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in ('L0_frame', 'L1_word')
        target = destination / relative
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for item in record['arrays']:
                offset, size, head, width = (item[k] for k in ('offset', 'bytes', 'header_bytes', 'itemsize'))
                raw = data[offset:offset + size]
                body = np.frombuffer(raw[head:], np.uint8).reshape(width, -1).T.tobytes()
                original = raw[:head] + body
                assert sha(original) == item['sha256'], (record['file'], item['name'])
                info = zipfile.ZipInfo(item['name'], date_time=tuple(item['date_time']))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = item['external_attr']
                info.create_system = item['create_system']
                with z.open(info, 'w', force_zip64=True) as stream:
                    stream.write(original)
        content = output.getvalue()
        assert sha(content) == record['sha256'], 'NPZ reconstruction differs: ' + record['file']
        if target.exists():
            assert target.read_bytes() == content, 'Refusing to overwrite different features: ' + str(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return {'files_restored': len(meta['files']), 'all_original_npz_sha256_match': True,
            'l0_files': 300, 'l1_files': 100, 'samples': 100}


def pack(source):
    files = sorted((source / 'L0_frame').glob('*.npz')) + sorted((source / 'L1_word').glob('*.npz'))
    assert len(files) == 400
    assert len(list((source / 'L0_frame').glob('*.npz'))) == 300
    payload, records = io.BytesIO(), []
    for path in files:
        record = {'file': path.relative_to(source).as_posix(), 'sha256': sha(path.read_bytes()), 'arrays': []}
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                data = z.read(info.filename)
                head, width = header(data)
                body = np.frombuffer(data[head:], np.uint8).reshape(-1, width).T.tobytes()
                record['arrays'].append({'name': info.filename, 'offset': payload.tell(), 'bytes': len(data),
                                         'header_bytes': head, 'itemsize': width, 'sha256': sha(data),
                                         'date_time': info.date_time, 'external_attr': info.external_attr,
                                         'create_system': info.create_system})
                payload.write(data[:head] + body)
        records.append(record)
    compressed = lzma.compress(payload.getvalue(), preset=6)
    meta = {'format': 'emomosaic-lossless-npy-byte-shuffle-xz-v1',
            'description': 'Complete original L0/L1 arrays; no quantization, truncation, or re-extraction.',
            'uncompressed_bytes': payload.tell(), 'payload_sha256': sha(compressed), 'files': records}
    (source / 'full_features.bin.xz').write_bytes(compressed)
    (source / 'full_features_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    with tempfile.TemporaryDirectory(prefix='emomosaic-p1-feature-roundtrip-') as temporary:
        result = restore(source, Path(temporary))
    result.update(payload_bytes=len(compressed), payload_sha256=sha(compressed), lossless=True)
    (source / 'full_features_verification.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'restore', 'verify'])
    parser.add_argument('--bundle', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == 'pack':
        result = pack(args.bundle)
    elif args.action == 'restore':
        result = restore(args.bundle, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='emomosaic-p1-feature-check-') as temporary:
            result = restore(args.bundle, Path(temporary))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
