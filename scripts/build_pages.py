"""Build the static GitHub Pages site with Python's standard library."""
from pathlib import Path
import html
import hashlib
import json
import re
import shutil
import string
from sync_site_i18n import sync

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website"
DEST = ROOT / "_site"
PAGES = [
    ('index', '研究概述', 'OVERVIEW', [('research', '研究任务'), ('findings', '实验结果'), ('application', '离线应用')]),
    ('problem1', '特征对齐', 'ALIGNMENT', [('analysis', '研究动机'), ('features', '特征表示'), ('alignment', '时序对齐'), ('findings', '结果分析'), ('sources', '结果文件')]),
    ('method', '模型方法', 'METHOD', [('pipeline', '研究动机'), ('architecture', '模型结构'), ('training', '双视图训练'), ('fusion', '可用度约束融合'), ('findings', '结果分析'), ('explanation', '预测解释')]),
    ('problem3', '预测解释', 'EXPLANATIONS', [('analysis', '研究动机'), ('attribution', '模态归因'), ('evidence', '局部证据'), ('findings', '结果分析'), ('case', '样本分析')]),
    ('results', '实验结果', 'RESULTS', [('evaluation', '总体性能'), ('comparison', '候选比较'), ('evidence-results', '解释保真度'), ('protocol', '评估设置')]),
    ('explore', '样本分析', 'SAMPLE ANALYSIS', [('playground', '样本预测'), ('attribution', '模态归因'), ('evidence', '局部证据')]),
    ('inference', '在线推理', 'INFERENCE', [('input', '输入数据'), ('prediction', '预测结果'), ('model', '模型与算法'), ('protocol', '输入协议')]),
    ('app', '离线应用', 'APPLICATION', [('interface', '应用界面'), ('capabilities', '输入模态'), ('deployment', '端侧部署')]),
    ('reproduce', '复现说明', 'REPRODUCTION', [('start', '运行环境'), ('resources', '研究资料'), ('structure', '项目结构')]),
]


def build():
    sync()
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir()
    for directory in ("assets", "data"):
        shutil.copytree(SOURCE / directory, DEST / directory)
    template = string.Template((SOURCE / "template.html").read_text(encoding="utf-8"))
    data = json.loads((SOURCE / "data/results.json").read_text(encoding="utf-8"))

    def version_assets(page):
        # Returning readers must not mix new math markup with old CSS/scripts.
        def replace(match):
            version = hashlib.sha256((SOURCE / match[2]).read_bytes()).hexdigest()[:12]
            return f'{match[1]}{match[2]}?v={version}{match[3]}'
        return re.sub(r'((?:src|href)=")(assets/[^"?]+\.(?:js|css))(")', replace, page)

    def metric(match):
        value = data
        for key in match.group(1).split("."):
            value = value[key]
        return f"{value:.4f}"

    for name, title, eyebrow, sections in PAGES:
        navigation_links = []
        for key, label, _, _ in PAGES:
            active = ' active' if key == name else ''
            current = ' aria-current="page"' if key == name else ''
            navigation_links.append(f'<a href="{key}.html" class="nav-link{active}"{current}>{label}</a>')
        navigation = ''.join(navigation_links)
        toc = "".join(f'<a href="#{key}">{label}</a>' for key, label in sections)
        content = (SOURCE / "pages" / f"{name}.html").read_text(encoding="utf-8")
        content = re.sub(r"@metric:([a-zA-Z0-9_.]+)@", metric, content)
        rendered = template.substitute(
            title=html.escape(title), name=name, eyebrow=eyebrow,
            navigation=navigation, toc=toc, content=content,
        )
        (DEST / f"{name}.html").write_text(version_assets(rendered), encoding="utf-8")
    (DEST / ".nojekyll").touch()
    (DEST / "404.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<base href="/EmoMosaic/"><script src="assets/i18n-en.js" defer></script><script src="assets/i18n.js" defer></script>'
        '<title>页面未找到 · EmoMosaic</title><body style="font-family:system-ui;max-width:40rem;margin:15vh auto;padding:2rem">'
        '<button data-language-toggle type="button" aria-label="Switch to English">English</button><h1>404 · 页面未找到</h1>'
        '<a href="/EmoMosaic/">返回 EmoMosaic →</a></body></html>', encoding="utf-8"
    )
    urls = '\n'.join(f'  <url><loc>https://rudykon.github.io/EmoMosaic/{name}.html</loc></url>'
                     for name, *_ in PAGES)
    (DEST / 'sitemap.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + '\n</urlset>\n', encoding='utf-8')
    (DEST / 'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: https://rudykon.github.io/EmoMosaic/sitemap.xml\n')
    data = json.loads((DEST / "data/results.json").read_text(encoding="utf-8"))
    print(f"Built {len(PAGES)} pages and {len(data['samples'])} replay samples in {DEST}")


if __name__ == "__main__":
    build()
