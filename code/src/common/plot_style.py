"""Conventional MATLAB-style figures with shared fonts and external legends."""
import json
import os
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib import font_manager
from matplotlib.text import Text

# Text / audio / vision keep the same colors throughout all three problems.
COLORS = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#A2142F', '#EDB120', '#4D4D4D']
# Neutral, opaque interval backgrounds; measured curves are drawn above them.
INTERVAL_COLORS = ['#D0D0D0', '#ECECEC']
INK = '#000000'
MUTED = '#404040'
FONT_FAMILIES = ['Times New Roman', 'SimSun']
POLARITY = {'Negative': COLORS[0], 'Neutral': COLORS[1], 'Positive': COLORS[2]}
SEQUENTIAL = LinearSegmentedColormap.from_list('paper_blue', ['#F2F2F2', '#9ECAE1', '#3182BD', '#08519C'])
DIVERGING = LinearSegmentedColormap.from_list('paper_blue_red', ['#2166AC', '#F7F7F7', '#B2182B'])


@lru_cache(maxsize=1)
def register_fonts():
    """Use locally installed fonts for experiment visualizations."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    latin = next((name for name in ['Times New Roman', 'TeX Gyre Termes',
                                   'Liberation Serif', 'DejaVu Serif'] if name in installed),
                 'DejaVu Serif')
    chinese = next((name for name in ['SimSun', 'Noto Serif CJK SC', 'Noto Sans CJK SC',
                                     'Source Han Serif SC', 'Source Han Sans SC']
                    if name in installed), None)
    FONT_FAMILIES[:] = [latin] + ([chinese] if chinese else [])


def apply_style():
    register_fonts()
    mpl.rcParams.update({
        'font.family': FONT_FAMILIES, 'font.serif': FONT_FAMILIES, 'font.size': 9,
        'axes.titlesize': 10, 'axes.titleweight': 'normal', 'axes.titlepad': 8,
        'axes.labelsize': 9, 'axes.labelpad': 6,
        'axes.titlecolor': INK,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'axes.linewidth': .8, 'axes.spines.top': True, 'axes.spines.right': True,
        'axes.grid': False, 'axes.axisbelow': True,
        'grid.color': '#BFBFBF', 'grid.alpha': 1, 'grid.linewidth': .5,
        'grid.linestyle': ':',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.prop_cycle': cycler(color=COLORS),
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'lines.solid_capstyle': 'butt', 'lines.dash_capstyle': 'butt',
        'lines.solid_joinstyle': 'miter', 'lines.dash_joinstyle': 'miter',
        'patch.linewidth': .5, 'patch.edgecolor': INK, 'patch.force_edgecolor': True,
        'legend.frameon': True, 'legend.fancybox': False, 'legend.fontsize': 8,
        'legend.facecolor': 'white', 'legend.edgecolor': INK, 'legend.framealpha': 1,
        'legend.borderpad': .4,
        'legend.labelcolor': INK,
        'legend.handlelength': 2.2, 'legend.columnspacing': 1.5,
        'legend.labelspacing': .65, 'legend.borderaxespad': 0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .12,
        'text.parse_math': False,
    })


def boxplot(ax, values, labels, colors=None):
    colors = colors or COLORS
    boxes = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True,
        widths=.52, boxprops={'linewidth': .9, 'joinstyle': 'miter'},
        medianprops={'color': INK, 'linewidth': 1.1, 'solid_capstyle': 'butt'},
        whiskerprops={'color': MUTED, 'linewidth': .8, 'linestyle': '--'},
        capprops={'color': MUTED, 'linewidth': .8, 'solid_capstyle': 'butt'})
    for i, patch in enumerate(boxes['boxes']):
        color = colors[i % len(colors)]
        patch.set_facecolor('white')
        patch.set_edgecolor(color)
    return boxes


def heatmap_text_color(image, value):
    r, g, b, _ = image.cmap(image.norm(value))
    return 'white' if .2126*r + .7152*g + .0722*b < .56 else INK


def external_legends(fig):
    """Reserve a shared strip below every plot, including its axis labels."""
    entries = {}
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    legends += list(fig.legends)
    for legend in legends:
        title = legend.get_title().get_text()
        for handle, label in zip(legend.legend_handles, legend.get_texts()):
            text = label.get_text()
            if title == 'Segments':
                text = f'Segments: {float(text):g}'
            entries.setdefault(text, handle)
        legend.remove()
    if not entries:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottoms = [ax.get_tightbbox(renderer).y0 for ax in fig.axes if ax.get_visible()]
    y = min(-.012, min(bottoms, default=0) / fig.bbox.height - .025)
    columns = len(entries) if len(entries) <= 5 else (3 if len(entries) == 6 else 4)
    fig.legend(list(entries.values()), list(entries), loc='upper center',
               bbox_to_anchor=(.5, y), ncol=columns, frameon=True, fancybox=False)


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        # Keep native rectangular bars/cells and unmodified source video pixels.
        if ax.images:
            ax.grid(False)
        for spine in ax.spines.values():
            spine.set_capstyle('butt')
            spine.set_joinstyle('miter')
    external_legends(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        for i, ax in enumerate(fig.axes):
            if ax.axison and box.overlaps(ax.get_tightbbox(renderer)):
                overlaps.append(i)
    if overlaps:
        raise RuntimeError(f'{name}: legend overlaps axes {overlaps}')
    text_artists = [artist for artist in fig.findobj(Text)
                    if artist.get_visible() and artist.get_text().strip()]
    if any(artist.get_fontfamily() != FONT_FAMILIES for artist in text_artists):
        raise RuntimeError(f'{name}: text does not use the paper font families')
    text_colors = [to_rgba(artist.get_color()) for artist in text_artists]
    if any(color not in (to_rgba(INK), to_rgba('white')) for color in text_colors):
        raise RuntimeError(f'{name}: text must be black (or white on a dark heatmap)')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')
    audit_dir = os.environ.get('EMOMOSAIC_STYLE_AUDIT_DIR')
    if audit_dir:
        target = Path(audit_dir)
        target.mkdir(parents=True, exist_ok=True)
        key = directory.parent.name + '__' + name
        (target / f'{key}.json').write_text(json.dumps({
            'figure': str(directory / f'{name}.pdf'),
            'axes': len(fig.axes), 'external_legends': len(fig.legends),
            'legend_overlaps': overlaps, 'png_dpi': 300,
            'font_families': FONT_FAMILIES,
            'style': 'matlab-plain',
            'series_colors': COLORS,
            'black_text_count': text_colors.count(to_rgba(INK)),
            'white_heatmap_text_count': text_colors.count(to_rgba('white')),
        }, indent=2))
    plt.close(fig)
