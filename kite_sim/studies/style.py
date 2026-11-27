"""
style.py - Axis style, class labels and notes shared by the figures
===================================================================
"""

from matplotlib.transforms import blended_transform_factory

from .. import wind as da
from .settings import BASE, C_GREY, S_MAX, SITE_ELEVATION, USE_SITE_ELEVATION


def short_label(cls):
    """'C3\\n5.5-6.0' style tick label for a class."""
    name = da.CLASS_NAMES[cls - 1]
    return name.split(': ')[0] + '\n' + name.split(': ')[1].replace(' m/s', '')


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(axis='y', color='#e5e5e5', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _mark_infeasible(ax, xs, rs, y_frac=0.04):
    """Write 'not feasible' above every class without a design (y in axes fraction)."""
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, r in zip(xs, rs):
        if r.get('no_data'):
            ax.text(x, y_frac, 'no data\npoints', ha='center', va='bottom',
                    fontsize=7, color=C_GREY, transform=tr)
        elif not r['feasible']:
            ax.text(x, y_frac, f'not\nfeasible\n(> {S_MAX:.0f} m²)', ha='center', va='bottom',
                    fontsize=7, color=C_GREY, transform=tr)


def _base_note():
    return (f"Baseline: heights {BASE['zmin']:.0f}-{BASE['zmax']:.0f} m, "
            f"F_max {BASE['Fmax'] / 1e3:.0f} kN, target {BASE['P_target'] / 1e3:.0f} kW mean, "
            + ("air density at each class's mean ground elevation"
               if USE_SITE_ELEVATION and any(SITE_ELEVATION.values()) else "air density at sea level"))
