"""Build deterministic submission archives and verify every member before publishing."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def write_verified_archive(destination, files, limit_bytes=50_000_000, compression=zipfile.ZIP_DEFLATED):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.TemporaryDirectory(prefix='.package-', dir=destination.parent) as temporary:
        archive = Path(temporary) / destination.name
        with zipfile.ZipFile(archive, 'w', compression, compresslevel=9) as output:
            def add(name, content):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                output.writestr(info, content, compress_type=compression, compresslevel=9)

            for name, path in sorted(files.items()):
                member = PurePosixPath(name)
                if member.is_absolute() or '..' in member.parts or name == 'PACKAGE_MANIFEST.json':
                    raise ValueError(f'Invalid archive member: {name}')
                if 'github_token' in name or any(part.startswith('.venv') for part in member.parts):
                    raise ValueError(f'Private or environment file in archive: {name}')
                content = Path(path).read_bytes()
                manifest[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                add(name, content)
            add('PACKAGE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
        with zipfile.ZipFile(archive) as output:
            assert output.testzip() is None
            assert set(output.namelist()) == set(manifest) | {'PACKAGE_MANIFEST.json'}
            for name, item in manifest.items():
                content = output.read(name)
                assert len(content) == item['bytes']
                assert hashlib.sha256(content).hexdigest() == item['sha256'], name
        size = archive.stat().st_size
        if size > limit_bytes:
            raise ValueError(f'{destination.name}: {size} bytes exceeds {limit_bytes}')
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.replace(destination)
    return dict(bytes=size, sha256=checksum, files=len(manifest)+1)
