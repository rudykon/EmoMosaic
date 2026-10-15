"""Validate the deployable site and frozen fusion snapshot using stdlib only."""
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from build_pages import PAGES

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / '_site'


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if 'id' in attributes:
            assert attributes['id'] not in self.ids, f"Duplicate id: {attributes['id']}"
            self.ids.add(attributes['id'])
        for key in ('href', 'src'):
            if attributes.get(key):
                self.links.append(attributes[key])
        if tag == 'img':
            assert 'alt' in attributes, 'Image is missing an alt attribute'


def close(actual, expected, tolerance=1e-8):
    assert math.isfinite(actual) and abs(actual - expected) < tolerance, (actual, expected)


def shapley(values, bit):
    return sum(math.factorial(mask.bit_count()) * math.factorial(2-mask.bit_count()) / 6
               * (values[mask | bit] - values[mask]) for mask in range(8) if not mask & bit)


def check_inference():
    folder = ROOT / 'assets/inference'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['schema'] == 'emomosaic_browser_inference_v1'
    assert manifest['model'] == 'original_rm_lora_fusion_bias'
    assert len(manifest['adapters']) == 5
    assert manifest['assetBaseUrl'] == './', 'Browser models must be served by this site'
    model_count = 0
    for name, meta in manifest['assets'].items():
        assert len(meta['sha256']) == 64 and meta['bytes'] > 0
        if name.startswith('models/'):
            model_count += 1
        path = (folder / name).resolve()
        assert path.is_relative_to(folder) and path.is_file()
        assert path.stat().st_size == meta['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256'], name
    assert model_count == 33
    assert sum(m['bytes'] for m in manifest['assets'].values()) == manifest['downloadBytes']
    sample_path = ROOT / manifest['sample']['path']
    assert sample_path.resolve().is_relative_to(ROOT / 'data/inference')
    assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == manifest['sample']['sha256']
    sample = json.loads(sample_path.read_text())
    assert sample['schema'] == 'emomosaic_aligned_50_v1' and sample['id'] == 'attachment4-19'
    assert set(sample) == {'schema', 'id', 'transcript', 'source', 'source_sha256', 'length',
                           'token_ids', 'observed', 'text', 'audio', 'vision'}, 'Inputs must not contain predictions'
    assert 1 <= sample['length'] <= 50 and len(sample['token_ids']) == 50
    assert len(sample['observed']) == 3
    assert all(len(row) == 50 and all(type(v) is bool for v in row) for row in sample['observed'])
    assert all(not any(row[sample['length']:]) for row in sample['observed'])
    for name, dim in [('text', 768), ('audio', 74), ('vision', 35)]:
        assert len(sample[name]) == 50 * dim and all(math.isfinite(v) for v in sample[name])
    print('PASS: local inference runtime and all 33 model hashes, input schema')


def check_project_links():
    documents = list(PROJECT.glob('README*.md')) + list((PROJECT / 'assets/readme').glob('*.md'))
    for directory in (PROJECT / 'website', ROOT):
        documents.extend(path for path in directory.rglob('*')
                         if path.suffix in {'.md', '.html', '.json', '.js', '.css', '.svg', '.txt'})
    for path in documents:
        assert 'EmoMosaic_GMCM2026' not in path.read_text(encoding='utf-8'), f'Legacy project reference: {path}'
    release = json.loads((ROOT / 'data/app-release.json').read_text())
    assert release['release_url'].startswith('https://github.com/rudykon/EmoMosaic/releases/tag/')
    assert release['apk']['url'].startswith('https://github.com/rudykon/EmoMosaic/releases/download/')
    for path in (PROJECT / 'README.md', PROJECT / 'README.zh-CN.md', ROOT / 'app.html'):
        content = path.read_text(encoding='utf-8')
        assert release['apk']['url'] in content and release['release_url'] in content, f'Missing release link: {path}'
    print('PASS: README and website contain no legacy project links; APK links use this repository')


def check():
    check_project_links()
    math_folder = ROOT / 'assets/vendor/katex'
    math_assets = json.loads((math_folder / 'sources.json').read_text())
    for name, metadata in math_assets['files'].items():
        asset = (math_folder / name).resolve()
        assert asset.is_relative_to(math_folder) and asset.stat().st_size == metadata['bytes']
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == metadata['sha256'], name
    for font in re.findall(r'url\(([^)]+)\)', (math_folder / 'katex.min.css').read_text()):
        assert (math_folder / font.strip('\"\' ')).is_file(), f'Missing math font: {font}'
    pages = {}
    for path in ROOT.glob('*.html'):
        text = path.read_text(encoding='utf-8')
        if path.name != '404.html':
            assert f'https://rudykon.github.io/EmoMosaic/{path.name}' in text
        assert '@metric:' not in text, f'Unrendered metric: {path.name}'
        assert not any(phrase in text for phrase in ('暂定最终', '历史阈值极性', '梯度高分词项'))
        parser = PageParser()
        parser.feed(text)
        pages[path] = parser
    assert len(pages) == len(PAGES) + 1, 'Build all pages and the 404 page first'
    for path, parser in pages.items():
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = ROOT / 'index.html' if url.path == '/EmoMosaic/' else (
                (path.parent / unquote(url.path)).resolve() if url.path else path)
            assert target.is_relative_to(ROOT), f'Link escapes site: {link}'
            assert target.exists(), f'Broken link in {path.name}: {link}'
            if url.fragment and target in pages:
                assert unquote(url.fragment) in pages[target].ids, f'Broken fragment: {link}'
    data = json.loads((ROOT / 'data/results.json').read_text(encoding='utf-8'))
    assert data['schema'] == 'emomosaic_website_fusion_v2'
    assert data['provenance']['model'] == 'original_rm_lora_fusion_bias'
    assert data['labels'] == ['Negative', 'Neutral', 'Positive']
    assert len(data['samples']) == 20
    assert {s['id'] for s in data['samples']} == {f'{i:02d}' for i in range(1,21)}
    for sample in data['samples']:
        subsets = sample['subsets']
        assert set(subsets) == set(map(str, range(8)))
        for row in subsets.values():
            assert math.isfinite(row['intensity']) and -3 <= row['intensity'] <= 3
            assert len(row['probabilities']) == 3
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in row['probabilities'])
            close(sum(row['probabilities']), 1)
        close(subsets['7']['intensity'], sample['prediction'], 2e-5)
        for p, expected in zip(sample['probabilities'], subsets['7']['probabilities']):
            close(p, expected, 2e-5)
        target = max(range(3), key=sample['probabilities'].__getitem__)
        assert target == sample['target_class']
        assert sample['polarity'] == data['labels'][target]
        for i, modality in enumerate(sample['modalities']):
            close(modality['shapley'], shapley([subsets[str(s)]['intensity'] for s in range(8)],1 << i))
            for c in range(3):
                close(modality['all_class_shapley'][c], shapley([subsets[str(s)]['probabilities'][c] for s in range(8)],1 << i))
            close(modality['class_shapley'], modality['all_class_shapley'][target])
            close(sum(modality['all_class_shapley']),0)
        close(sum(m['gate'] for m in sample['modalities']),1,2e-6)
        close(sum(m['contribution'] for m in sample['modalities']) + sample['interaction'] + sample['expert_contribution'], sample['prediction'],2e-5)
        for target, rows in sample['evidence'].items():
            assert len(rows) <= 3
            scores = [r['class_probability_delta'] if target=='classification' else abs(r['intensity_delta']) for r in rows]
            assert scores == sorted(scores, reverse=True)
            for row in rows:
                assert 0 <= row['char_start'] < row['char_end'] <= len(sample['text'])
                if row['timing_usable']:
                    assert 0 <= row['time_start'] <= row['time_end']
    for split, count in (('test',727), ('valid',728)):
        for condition in ('clean','missing'):
            metrics=data['metrics'][split][condition]
            cm=metrics['confusion']
            assert sum(map(sum,cm)) == count == metrics['n']
            close(sum(cm[i][i] for i in range(3))/count,metrics['accuracy'])
            class_f1=[2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in range(3)]
            close(sum(class_f1)/3,metrics['macro_f1'])
            close(sum(class_f1[i]*sum(cm[i]) for i in range(3))/count,metrics['weighted_f1'])
            final=next(row for row in data['comparison'] if row['model']==data['provenance']['model'])
            for key in ('accuracy','macro_f1','mae','corr'):
                close(final['metrics'][split][condition][key],metrics[key])
    close(data['test_mean_f1']['new'],sum(data['metrics']['test'][c]['macro_f1'] for c in ('clean','missing'))/2)
    assert data['app']['version'] == '2.2.1'
    assert data['app']['physical_android_test'] is False
    figures = json.loads((ROOT / 'assets/paper/sources.json').read_text(encoding='utf-8'))['figures']
    for figure in figures:
        asset = ROOT / figure['file']
        assert asset.is_relative_to(ROOT / 'assets/paper'), 'Figure path escapes paper assets'
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == figure['sha256'], f"Paper figure changed: {asset.name}"
        original = ROOT / figure['pdf']
        assert original.is_relative_to(ROOT / 'assets/paper'), 'PDF path escapes paper assets'
        assert hashlib.sha256(original.read_bytes()).hexdigest() == figure['source_sha256'], f"Original PDF changed: {original.name}"
    assert (ROOT / '.nojekyll').exists()
    check_inference()
    print(f'PASS: {len(PAGES)} pages, {len(figures)} paper figures, local links, 160 subset predictions, intensity/all-class Shapley, LoRA decomposition, evidence and four confusion matrices')


if __name__ == '__main__':
    check()
