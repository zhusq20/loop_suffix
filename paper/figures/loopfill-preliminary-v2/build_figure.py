"""Render per-loop and additional hit rates together from the frozen experiment.

Run this file directly; figure-data.json is a compact, verified data snapshot.
No inference or new sampling is performed here.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def intervals(ax, x, ci):
    ax.vlines(x, ci[:, 0], ci[:, 1], color='#333333', linewidth=.8)
    for pos, (lo, hi) in zip(x, ci):
        ax.hlines([lo, hi], pos - .07, pos + .07, color='#333333', linewidth=.8)


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / 'figure-data.json').read_text())
    metrics = data['metrics']
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 2.05), layout='constrained')
    for key, depths, label, color, marker, style in [
        ('agreement', [1, 2, 3], 'Single-loop hit', '#3477a3', 'o', '-'),
        ('gains', [2, 3], 'Additional hit', '#d6782a', 's', '--'),
    ]:
        indices = np.asarray(depths) - 1
        values = np.asarray(metrics[key]['mean'])[indices] * 100
        bounds = np.asarray(metrics[key]['ci95'])[indices] * 100
        axes[0].plot(depths, values, color=color, marker=marker, linestyle=style,
                     markersize=3.5, linewidth=1, label=label)
        axes[0].fill_between(depths, bounds[:, 0], bounds[:, 1], color=color, alpha=.18)
        for depth, value, (lo, hi) in zip(depths, values, bounds):
            axes[0].vlines(depth, lo, hi, color=color, linewidth=.8)
            y = lo - 4 if key == 'agreement' else hi + 3
            axes[0].text(depth, y, f'{value:.2f}%', ha='left' if depth == 1 else 'center',
                         va='top' if key == 'agreement' else 'bottom', fontsize=6.5,
                         color=color)
    axes[0].set(xticks=[1, 2, 3], xlabel='Loop depth', xlim=(.65, 3.35),
                ylabel='Hit rate (%)', ylim=(0, 105), yticks=[0, 20, 40, 60, 80, 100],
                title='(a) Loop hit rates')
    axes[0].legend(loc='center', frameon=False, fontsize=6.5, handlelength=1.5,
                   handletextpad=.5)
    means = np.asarray(metrics['lengths']['mean'])
    ci = np.asarray(metrics['lengths']['ci95'])
    axes[1].bar(range(3), means, color=['#a9b4bd', '#609bbf', '#e4a366'], width=.65)
    intervals(axes[1], range(3), ci)
    for i, value in enumerate(means):
        axes[1].text(i, ci[i, 1] + .04, f'{value:.2f}', ha='center', va='bottom', fontsize=7)
    axes[1].set(xticks=range(3), xticklabels=['Suffix', '+Shallow', '+Requery'],
                ylabel='Correct draft length', title='(b) Suffix synergy', ylim=(0, 2.8))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(directory / f'preliminary-hit-rates.{extension}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
