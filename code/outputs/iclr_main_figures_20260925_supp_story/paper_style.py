"""Canonical ToM manuscript figure style, v1 (2026-09-21)."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import textwrap

COLORS = dict(initial='#4A7298', intervention='#C07754', mixed='#A66F90',
              supplied='#528875', constant='#68737D', frequency='#9A7B4A')
DIVERGING = LinearSegmentedColormap.from_list(
    'payoff_muted', ['#8F614F', '#F8F7F2', '#347C78'], N=256)

def apply():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'grid.alpha': .15, 'grid.linewidth': .5,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.unicode_minus': True,
    })

def panel(ax, letter, title):
    ax.set_title(f'{letter}  '+textwrap.fill(title, width=25 if letter!='c' or len(title)<45 else 72), loc='left', fontweight='normal', pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis='y')

def legend(ax, **kwargs):
    return ax.legend(frameon=False, handlelength=1.3, handletextpad=.5,
                     columnspacing=.8, **kwargs)

def layout(fig):
    fig.set_constrained_layout_pads(w_pad=.055, h_pad=.065, wspace=.08, hspace=.12)
