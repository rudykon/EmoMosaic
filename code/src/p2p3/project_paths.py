"""Location-independent paths for the standalone COLING project.

ROOT intentionally denotes code/ for compatibility with frozen model records.
No resource is resolved through the parent research workspace.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / 'data'
PAPER_ROOT = PROJECT_ROOT / 'paper'


def resolve_asset(value):
    """Resolve local assets, including paths in historical provenance records."""
    path = Path(value)
    text = path.as_posix()
    if text.startswith('<PROJECT_ROOT>/'):
        text = text[len('<PROJECT_ROOT>/'):]
    if path.is_absolute():
        try:
            path.relative_to(PROJECT_ROOT)
            return path
        except ValueError:
            # Historical absolute paths are metadata, never a parent fallback.
            anchors = [text.rfind('/' + anchor)
                       for anchor in ('data/', 'models/', 'outputs/', 'src/', 'paper/')]
            position = max(anchors)
            if position < 0:
                raise ValueError(f'Asset is outside this project: {value}')
            # /mnt/data is a mount prefix, not the old project's data directory.
            # The rightmost resource anchor identifies the actual project asset.
            text = text[position + 1:]
    for old, new in {
        '原始数据': 'source_data',
        '附件1-数据集原始多模态样本': 'attachment1_raw_multimodal',
        'MOSEI数据集部分原始视频-100条': 'mosei_raw_videos_100',
        '附件2-数据集特征文件': 'attachment2_features',
        '附件3-模态缺失特征样本': 'attachment3_missing_modalities',
        '附件4-可解释专项视频样本与特征文件': 'attachment4_interpretability',
        '非对齐版本': 'unaligned',
        '未对齐版本': 'unaligned',
        '对齐版本': 'aligned',
    }.items():
        text = text.replace(old, new)
    text = text.replace('src/P2P3_model/', 'src/p2p3/')
    if text.startswith(('data/', 'paper/')):
        return PROJECT_ROOT / text
    return ROOT / text


def relative_asset(value):
    """Keep provenance paths relative to code/, including ../data references."""
    return Path(os.path.relpath(Path(value), ROOT)).as_posix()


def python_executable(kind='default'):
    """Use an explicit interpreter, a local environment, or the active Python."""
    variable = {'alignment': 'COLING_ALIGNMENT_PYTHON',
                'qwen35': 'COLING_QWEN35_PYTHON'}.get(kind)
    if kind == 'alignment' and not os.environ.get(variable) and os.environ.get('P1_PYTHON'):
        variable = 'P1_PYTHON'
    if variable and os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{variable} does not identify a Python executable: {path}')
        return str(path)
    environment = {'alignment': '.venv-p1', 'qwen35': '.venv-qwen35'}.get(kind)
    if environment:
        for base in (PROJECT_ROOT, ROOT):
            path = base / environment / 'bin/python'
            if path.is_file():
                return str(path)
    return sys.executable
