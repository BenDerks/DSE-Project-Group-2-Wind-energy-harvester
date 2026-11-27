"""
common.py - Settings and helpers shared by the verification cases
=================================================================

SUMMARY collects one row per compared quantity (record()); run_verification.py saves it.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import kite_model as km                     # noqa: E402


OUT = HERE.parent / 'outputs' / 'verification'
V_GRID = np.arange(1.0, 25.01, 0.5)

# First version of kite_model (before this verification): the current code with the three
# options added during the verification switched off
FIRST_VERSION = dict(transition_energy=False, depower=False, optimise_elevation=False)

C_REF = '#3a3a3a'
C_NEW = '#2b6cb0'
C_OLD = '#9ecae1'
C_ALT = '#d97706'
C_GREY = '#9a9a9a'
SUMMARY = []


# =====================================================================================
# HELPERS
# =====================================================================================
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


def _note(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=C_GREY, wrap=True)


def record(reference, quantity, ref_value, model_value, unit, note=''):
    """Add one compared quantity to the summary table."""
    diff = 100 * (model_value / ref_value - 1) if ref_value else np.nan
    SUMMARY.append(dict(reference=reference, quantity=quantity, unit=unit,
                        reference_value=round(ref_value, 3), model_value=round(model_value, 3),
                        difference_pct=round(diff, 1), note=note))


def model_curve(vs, S, m, zmin, zmax, Fmax, p, eta=None, geo_update=None):
    """
    Power curve of kite_model at the wind speeds vs.

    Inputs: S [m2] projected area, m [kg] airborne mass, zmin/zmax [m], Fmax [N], p parameters,
        eta drivetrain efficiency (None = mechanical power),
        geo_update values that override geometry() (e.g. known tether diameter, air density).
    Output: cycle power [W] per wind speed (0 below cut-in / above cut-out).
    """
    p = dict(p, w=m / S)
    geo = km.geometry(zmin, zmax, Fmax, p)
    if geo_update:
        geo.update(geo_update)
    out = []
    for v in vs:
        op = km.operating_point(v, S, geo, Fmax, p)
        if not isinstance(op, dict):
            out.append(0.0)
            continue
        if eta is None:
            out.append(op['Pc'])
            continue
        dl_o = max(geo['dl'] - op.get('dlx', 0.0), 0.0)
        E_o = op['Po'] * dl_o / op['vo'] + op.get('Ex', 0.0)
        T = op['to'] + op['ti'] + op['tx']
        out.append((eta * E_o - op['Ei'] / eta) / T)
    return np.array(out)


def rayleigh(mean, vs):
    """Bin probabilities of a Rayleigh distribution (Weibull k = 2) on the grid vs."""
    dv = vs[1] - vs[0]
    pj = km.weibull_bins(mean, 2.0, vs, dv)
    return pj
