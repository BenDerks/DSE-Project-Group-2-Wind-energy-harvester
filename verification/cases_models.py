"""
cases_models.py - Verification against independent model code and measured cycles
=================================================================================

1 three_regime   massless three-regime model of Luchsinger (2013), code of R. Schmehl
2 vdv2019        published model results and measurements of van der Vlugt et al. (2019)
"""

import sys
from math import radians, sin
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import kite_model as km                     # noqa: E402
import three_regime as tr                   # noqa: E402
import vdv2019_cases as vdv                 # noqa: E402
from common import (C_ALT, C_NEW, C_OLD, C_REF, FIRST_VERSION, OUT, V_GRID,  # noqa: E402
                    _note, _style, model_curve, record)


# =====================================================================================
# 1. THREE-REGIME REFERENCE CODE
# =====================================================================================
def case_three_regime():
    """
    kite_model against the massless three-regime model, 25 m2 TU Delft kite (kite_model.P,
    i.e. Schelbergen et al. 2020, Table 4).

    Model runs:
      massless, matched  no mass, azimuth 0, course 90 deg, no acceleration time, no
                         transition energy: as close to the reference as the structure allows
                         (reel-in and transition still differ: see three_regime.py)
      full model         as used in the design studies (mass, gravity, angles)
      first version      full model with FIRST_VERSION options
    """
    p = km.P
    S, Fmax = 19.75, 5000.0
    r_min, r_max = 200.0, 385.0
    beta = p['beta_o']
    zmin, zmax = r_min * sin(radians(beta)), r_max * sin(radians(beta))
    geo = km.geometry(zmin, zmax, Fmax, p)
    P_gen = 20e3           # [W] the "20 kW kite power system" of van der Vlugt et al. (2019)
    p = dict(p, P_gen_max=P_gen)
    ref = tr.power_curve(V_GRID, geo['rho'], S, p['CLo'], p['CDko'], p['CLi'], p['CDki'],
                         p['CDc'], geo['dt'], r_min, r_max, beta, Fmax, P_gen,
                         p['vmax'], p['vmax'])
    p_matched = dict(p, phi_o=0.0, chi_o=90.0, amax=1e9, rho_t=1e-9, transition_energy=False)
    matched = model_curve(V_GRID, S, 1e-6, zmin, zmax, Fmax, p_matched)
    m_kite = 22.8          # [kg] airborne mass of this kite, Schelbergen et al. 2020, Table 4
    full = model_curve(V_GRID, S, m_kite, zmin, zmax, Fmax, p)
    old = model_curve(V_GRID, S, m_kite, zmin, zmax, Fmax, dict(p, **FIRST_VERSION))

    # Reel-out power check: same force law -> identical reel-out power at the same f
    f_test = np.linspace(0.05, 0.6, 12)
    dev = []
    for v in (6.0, 10.0, 15.0):
        for f in f_test:
            s = km.state(v, f, S, 0.0, 0.0, p['CLo'], p['CDko'], radians(90 - beta), 0.0,
                         radians(90), geo['rho'], p)
            E2 = (p['CLo'] / p['CDko'])**2
            F_ref = 0.5 * geo['rho'] * v**2 * S * p['CLo'] * np.sqrt(1 + 1 / E2) * (1 + E2) \
                * (np.cos(radians(beta)) - f)**2
            dev.append(abs(s['Ftg'] / F_ref - 1))
    record('Three-regime code (Luchsinger 2013)', 'max. deviation of massless reel-out force, '
           '36 states', 1.0, 1.0 + max(dev), '-', 'identical force law: deviation is rounding only')

    # Reel-in force check: at the reference's steady reel-in elevation (lam = 0, no sideways
    # motion) kite_model.state must give the reference's reel-in force
    E2i = (p['CLi'] / p['CDki'])**2
    CRi = np.hypot(p['CLi'], p['CDki'])
    dev_in = []
    for v in (6.0, 10.0, 15.0):
        for f in np.linspace(-0.9, -0.1, 9):
            beta_in = np.arccos((np.sqrt(1 + E2i * (1 - f**2)) + f * E2i) / (1 + E2i))
            s = km.state(v, f, S, 0.0, 0.0, p['CLi'], p['CDki'], np.pi / 2 - beta_in, 0.0,
                         np.pi, geo['rho'], p)
            F_ref = 0.5 * geo['rho'] * v**2 * S * CRi * (np.sqrt(1 + E2i * (1 - f**2)) - f)**2 / (1 + E2i)
            dev_in.append(abs(s['Ftg'] / F_ref - 1))
    record('Three-regime code (Luchsinger 2013)', 'max. deviation of massless reel-in force at the '
           'steady reel-in elevation, 27 states', 1.0, 1.0 + max(dev_in), '-',
           'identical force law: deviation is rounding only')

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(V_GRID, ref['P_c'] / 1e3, color=C_REF, lw=2.5, label='three-regime reference code (massless)')
    ax.plot(V_GRID, matched / 1e3, color=C_ALT, lw=1.8, ls='--',
            label='kite_model, massless, angles and limits matched')
    ax.plot(V_GRID, full / 1e3, color=C_NEW, lw=2, label='kite_model as used (mass, gravity, angles)')
    ax.plot(V_GRID, old / 1e3, color=C_OLD, lw=1.5, label='kite_model, first version')
    _style(ax, 'Cycle power: kite_model against an independent reference code',
           'wind speed at the average flying height [m/s]', 'mechanical cycle power [kW]')
    ax.set_xlim(0, 25)
    ax.legend(frameon=False, fontsize=8)
    _note(fig, f'25 m² TU Delft kite (S = {S} m², {m_kite} kg, C_L/C_D: Schelbergen et al. 2020), F_max {Fmax / 1e3:.0f} kN, '
               f'generator limit {P_gen / 1e3:.0f} kW (reel-out), '
               f'tether {r_min:.0f}-{r_max:.0f} m, elevation {beta:.0f}°. The reference has no transition, '
               'no acceleration time and reels in at its steady reel-in elevation (no climb); above the force '
               'limit it lets the reel-out speed exceed v_max. Force laws of both phases agree exactly (summary.csv).')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / 'three_regime.png', dpi=150)

    # Upper bound: Loyd's ideal crosswind power for the same kite (massless, no reel-in),
    # 4/27 q v S C_R (1 + E^2) cos^3(beta) (Loyd 1980, with the reference code's elevation
    # correction). The model must never exceed it.
    E2 = (p['CLo'] / p['CDko'])**2
    P_loyd = 4 / 27 * 0.5 * geo['rho'] * V_GRID**3 * S * p['CLo'] * np.sqrt(1 + 1 / E2) \
        * (1 + E2) * np.cos(radians(beta))**3
    record('Loyd (1980) ideal crosswind power', 'max. ratio model / Loyd over all wind speeds',
           1.0, float(np.max(full / P_loyd)), '-', 'must stay below 1')

    for v in (8.0, 12.0):
        i = int(np.argmin(abs(V_GRID - v)))
        record('Three-regime code (Luchsinger 2013)', f'cycle power at {v:.0f} m/s, matched massless',
               ref['P_c'][i] / 1e3, matched[i] / 1e3, 'kW',
               'reference: kite already at its steady reel-in elevation, no transition; '
               'kite_model: kite climbs from 25 deg during reel-in, then flies back down')
        record('Three-regime code (Luchsinger 2013)', f'cycle power at {v:.0f} m/s, full model',
               ref['P_c'][i] / 1e3, full[i] / 1e3, 'kW', 'includes mass and the 13 deg azimuth')
    return dict(ref=ref, matched=matched, full=full, old=old)


# =====================================================================================
# 2. VAN DER VLUGT ET AL. (2019): MODEL RESULTS AND MEASUREMENTS
# =====================================================================================
def case_vdv2019():
    """Phase powers and durations of the two measured cycles."""
    rows = {}
    for name in ('moderate', 'strong'):
        rows[name] = dict(vdv.PUBLISHED[name])
        rows[name]['replica'] = vdv.replica(name)
        rows[name]['as used'] = vdv.kite_model_as_used(name)
        p_old = vdv.params
        vdv.params = lambda c, _f=p_old: dict(_f(c), **FIRST_VERSION)
        rows[name]['first version'] = vdv.kite_model_as_used(name)
        vdv.params = p_old

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [('model', 'published model (gravity incl.)', '#bdbdbd'),
              ('experiment', 'measurement', C_REF),
              ('replica', 'kite_model.state, paper set-up', C_ALT),
              ('as used', 'kite_model as used (one wind speed)', C_NEW),
              ('first version', 'kite_model first version', C_OLD)]
    phases = [('P_i', 'reel-in'), ('P_x', 'transition'), ('P_o', 'reel-out'), ('P_c', 'cycle')]
    for ax, name in zip(axs, ('moderate', 'strong')):
        x = np.arange(len(phases))
        w = 0.8 / len(labels)
        for j, (key, lab, col) in enumerate(labels):
            vals = [rows[name][key][k] for k, _ in phases]
            ax.bar(x + (j - (len(labels) - 1) / 2) * w, vals, w, color=col, label=lab)
        ax.axhline(0, color='#555555', lw=0.8)
        ax.set_xticks(x, [ph for _, ph in phases])
        c = vdv.CASES[name]
        _style(ax, f'{name.capitalize()} wind ({c["v_mt"]} m/s at {c["v_mt"] and ""}z_mt)',
               ylabel='phase mean power [kW]')
        ax.set_title(f'{name.capitalize()} wind: {c["v_mt"]} m/s at the traction altitude, '
                     f'kite {c["A"] if "A" in c else ""}{c["S"]} m² projected', loc='left',
                     fontsize=10, fontweight='bold')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.04))
    _note(fig, 'Van der Vlugt et al. (2019), Tables 1-3: TU Delft 20 kW system, two measured cycles. '
               'Reel-in is negative (consumed). "Paper set-up": force-controlled phases and a logarithmic '
               'wind profile as in the paper, with kite_model.state() as the force balance.')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / 'vdv2019.png', dpi=150)

    for name in ('moderate', 'strong'):
        for key, lab in (('experiment', 'measured'), ('model', 'published model')):
            ref = rows[name][key]
            for q, unit in (('P_c', 'kW'), ('P_o', 'kW'), ('P_i', 'kW'), ('t_c', 's')):
                record(f'van der Vlugt 2019, {name} wind, {lab}', f'{q} (kite_model as used)',
                       ref[q], rows[name]['as used'][q], unit)
                record(f'van der Vlugt 2019, {name} wind, {lab}', f'{q} (first version)',
                       ref[q], rows[name]['first version'][q], unit)
            record(f'van der Vlugt 2019, {name} wind, {lab}', 'P_c (kite_model.state, paper set-up)',
                   ref['P_c'], rows[name]['replica']['P_c'], 'kW')
    return rows
