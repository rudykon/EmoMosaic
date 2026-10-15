"""Export the current manuscript figures for the website (requires pdftoppm)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    'fig_01_model_overview',
    'fig_02_explanation_fidelity',
    'fig_03_error_diagnostics',
    'fig_04_core_missingness_rate',
    'fig_05_core_span_position',
    'fig_06_core_span_length_reconstruction',
    'fig_07_missingness_pattern_replay',
    'fig_08_word_alignment_example',
    'fig_09_alignment_quality_summary',
    'fig_10_sample_b19_class_attribution',
    'fig_11_sample_b19_intensity_attribution',
    'fig_12_unlabeled_set_a_predictions',
)


def export(source):
    if not shutil.which('pdftoppm'):
        raise SystemExit('Install Poppler (pdftoppm) to export figures; site builds do not need it.')
    for name in FIGURES:
        if not (source / f'{name}.pdf').is_file():
            raise SystemExit(f'Missing manuscript figure: {source / (name + ".pdf")}')
    target = ROOT / 'website/assets/paper'
    target.mkdir(parents=True, exist_ok=True)
    figures = []
    for number, name in enumerate(FIGURES, 1):
        original = source / f'{name}.pdf'
        pdf = target / original.name
        shutil.copyfile(original, pdf)
        subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '2400', '-png',
                        str(pdf), str(target / name)], check=True, capture_output=True)
        png = target / f'{name}.png'
        width, height = struct.unpack('>II', png.read_bytes()[16:24])
        figures.append({
            'file': f'assets/paper/{png.name}',
            'pdf': f'assets/paper/{pdf.name}',
            'source': f'paper/figures/{original.name}',
            'sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'paper_figure': str(number),
            'width': width,
            'height': height,
        })
    manifest = {
        'source': 'Current EmoMosaic manuscript, paper/figures (Figures 1–12).',
        'preview': 'Complete PDF page rendered by pdftoppm, longest edge 2400 px; no cropping or redrawing.',
        'figures': figures,
    }
    (target / 'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(figures)} manuscript figures with original PDFs and PNG previews.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'paper/figures')
    export(parser.parse_args().source)
