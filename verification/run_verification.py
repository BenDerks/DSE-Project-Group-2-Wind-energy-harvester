"""
run_verification.py - Verify kite_model.py against independent code, test data and products
==========================================================================================

WHAT THIS FILE DOES
    Runs kite_model.py on four kinds of reference, each with the reference's OWN inputs, and
    makes one figure per reference plus a summary table:

    1 three_regime   independent reference CODE: the massless three-regime model of
                     Luchsinger (2013) as implemented by R. Schmehl (verification/three_regime.py)
    2 vdv2019        published MODEL RESULTS and MEASUREMENTS of two pumping cycles of the
                     TU Delft 20 kW system (van der Vlugt et al. 2019; verification/vdv2019_cases.py)
    3 falcon         PRODUCT: Kitepower Falcon 100 kW power curve (electrical)
    4 skysails       PRODUCT: SkySails PN-14 annual energy and rated power (wind at 200 m)

    Each case can also be run with the first version of the model (no energy in the
    transition), to show what the changes made during the verification did.

HOW TO RUN
    python verification/run_verification.py            all cases (about 10 minutes)
    python verification/run_verification.py falcon     one case

OUTPUT
    outputs/verification/*.png      figures
    outputs/verification/summary.csv one row per compared quantity: reference, model, difference
"""

import sys
import time
from math import radians, sin, log
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import kite_model as km
import three_regime as tr
import vdv2019_cases as vdv
from references import FALCON, SKYSAILS, ETA, falcon_curve

OUT = HERE.parent / 'outputs' / 'verification'
V_GRID = np.arange(1.0, 25.01, 0.5)

# The first version of kite_model (before this verification) is the current code with the
# three options that were added during the verification switched off
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

    Inputs: S [m2] projected area, m [kg] airborne mass, zmin/zmax [m], Fmax [N], p parameter
    dict, eta drivetrain efficiency (None = mechanical power), geo_update dict of values that
    replace the ones geometry() derives (e.g. a known tether diameter or air density).
    Output: array of cycle power [W] (0 below cut-in / above cut-out).
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


# =====================================================================================
# 1. THREE-REGIME REFERENCE CODE
# =====================================================================================
def case_three_regime():
    """
    kite_model against the massless three-regime model, for the 25 m2 TU Delft kite
    (parameters of kite_model.P, i.e. Schelbergen et al. 2020, Table 4).

    Three model runs:
      massless, matched  kite_model with no mass, azimuth 0, course 90 deg, no acceleration
                         time and no transition energy: as close to the reference as the
                         structure allows (reel-in and transition are still modelled
                         differently: see three_regime.py)
      full model         kite_model as used in the design studies (mass, gravity, angles)
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

    # Reel-in force check: at the steady reel-in elevation of the reference (kite not moving
    # sideways, lam = 0), kite_model.state must give the reference's reel-in force
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

    # Upper bound: the model may never exceed Loyd's ideal crosswind power with the same kite
    # (massless, no reel-in), 4/27 q v S C_R (1 + E^2) cos^3(beta) (Loyd 1980, with the
    # elevation correction of the reference code)
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


# =====================================================================================
# 3. KITEPOWER FALCON
# =====================================================================================
def _falcon_variants():
    """Kite variants: flat area [m2] -> (projected area [m2], mass [kg])."""
    m_per_flat = FALCON['mass_60'] / 60.0       # ASSUMED: mass scales with area (73 kg for 60 m2)
    out = {}
    for A, S in zip(FALCON['area_flat_V9'], FALCON['area_proj_V9']):
        out[A] = (S, m_per_flat * A)
    A = FALCON['curve_area_flat']
    out[A] = (A * FALCON['area_proj_V9'][0] / FALCON['area_flat_V9'][0], m_per_flat * A)
    return out


def case_falcon():
    """Electrical power curve of the Falcon against the model with the brochure's inputs."""
    beta = km.P['beta_o']
    zmin = FALCON['r_min'] * sin(radians(beta))
    zmax = FALCON['r_max'] * sin(radians(beta))
    zbar = 0.5 * (zmin + zmax)
    Fmax = FALCON['F_max']
    p = dict(km.P, P_gen_max=FALCON['P_reel_out_el'] / ETA)
    curves = {}
    for A, (S, m) in _falcon_variants().items():
        curves[A] = model_curve(V_GRID, S, m, zmin, zmax, Fmax, p, eta=ETA)
    A_ref = FALCON['curve_area_flat']
    S_ref, m_ref = _falcon_variants()[A_ref]
    old = model_curve(V_GRID, S_ref, m_ref, zmin, zmax, Fmax, dict(p, **FIRST_VERSION), eta=ETA)

    # Scenario: the reference wind speed is measured at 6 m (the ground-station anemometer
    # height of the TU Delft/Kitepower system, van der Vlugt et al. 2019, Table 1) with a
    # logarithmic profile, z0 = 0.07 m (same source). Then the wind at the kite is higher.
    z_ref, z0 = 6.0, 0.07
    scale_6m = log(z_ref / z0) / log(zbar / z0)      # v_6m = v_zbar * scale

    # Which kite L/D would match the published mean power at 7 m/s, for each wind height?
    LDs = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    vv = np.arange(0.25, 30, 0.5)
    ref7 = np.sum(rayleigh(7.0, vv) * falcon_curve(vv)) / 1e3
    m7_zbar, m7_6m = [], []
    for LD in LDs:
        P_LD = model_curve(V_GRID, S_ref, m_ref, zmin, zmax, Fmax, dict(p, CDko=p['CLo'] / LD), eta=ETA)
        m7_zbar.append(np.sum(rayleigh(7.0, V_GRID) * P_LD) / 1e3)
        m7_6m.append(np.sum(rayleigh(7.0, vv) * np.interp(vv, V_GRID * scale_6m, P_LD)) / 1e3)
    m7_zbar, m7_6m = np.array(m7_zbar), np.array(m7_6m)
    LD_zbar = float(np.interp(ref7, m7_zbar, LDs)) if m7_zbar.max() >= ref7 else np.nan
    LD_6m = float(np.interp(ref7, m7_6m, LDs)) if m7_6m.max() >= ref7 else np.nan

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    v = np.linspace(0, 25, 251)
    a1.plot(v, falcon_curve(v) / 1e3, color=C_REF, lw=2.5, label=f'Falcon, published ({A_ref:.0f} m², electrical)')
    cols = {60.0: C_OLD, 80.0: '#4292c6', 100.0: C_NEW}
    for A, P in curves.items():
        a1.plot(V_GRID, P / 1e3, color=cols.get(A, C_NEW), lw=2 if A == A_ref else 1.4,
                label=f'kite_model, {A:.0f} m² flat ({_falcon_variants()[A][0]:.0f} m² projected)')
    a1.plot(V_GRID, old / 1e3, color=C_GREY, lw=1, ls=':', label=f'kite_model first version, {A_ref:.0f} m²')
    a1.plot(V_GRID * scale_6m, curves[A_ref] / 1e3, color=C_ALT, lw=1.8, ls='--',
            label=f'kite_model {A_ref:.0f} m², if the published wind is at 6 m')
    a1.axhline(0, color='#555555', lw=0.8)
    a1.set_xlim(0, 25)
    _style(a1, 'Kitepower Falcon: power curve', 'wind speed [m/s] (model: at the average flying height)',
           'electrical cycle power [kW]')
    a1.legend(frameon=False, fontsize=7.5, loc='lower right')

    # Mean power for three Rayleigh mean wind speeds
    means = [5.0, 7.0, 9.0]
    vv = np.arange(0.25, 30, 0.5)
    ref_mean = [np.sum(rayleigh(mn, vv) * falcon_curve(vv)) / 1e3 for mn in means]
    mod_mean = [np.sum(rayleigh(mn, V_GRID) * curves[A_ref]) / 1e3 for mn in means]
    mod_6m = [np.sum(rayleigh(mn, vv) * np.interp(vv, V_GRID * scale_6m, curves[A_ref])) / 1e3
              for mn in means]
    x = np.arange(len(means))
    a2.bar(x - 0.27, ref_mean, 0.27, color=C_REF, label='published curve')
    a2.bar(x, mod_mean, 0.27, color=C_NEW, label='kite_model (wind at flying height)')
    a2.bar(x + 0.27, mod_6m, 0.27, color=C_ALT, label='kite_model (if published wind is at 6 m)')
    for xi, r, m1, m2 in zip(x, ref_mean, mod_mean, mod_6m):
        a2.text(xi - 0.27, r + 1, f'{r:.0f}', ha='center', fontsize=7.5)
        a2.text(xi, m1 + 1, f'{m1:.0f}\n{100 * (m1 / r - 1):+.0f}%', ha='center', fontsize=7.5)
        a2.text(xi + 0.27, m2 + 1, f'{m2:.0f}\n{100 * (m2 / r - 1):+.0f}%', ha='center', fontsize=7.5)
    a2.set_xticks(x, [f'mean {mn:.0f} m/s' for mn in means])
    a2.set_ylim(0, 1.35 * max(max(ref_mean), max(mod_6m), max(mod_mean)))
    _style(a2, 'Mean power (Rayleigh wind)', ylabel='mean electrical power [kW]')
    a2.legend(frameon=False, fontsize=7.5, loc='upper left')

    a3.plot(LDs, m7_zbar, 'o-', color=C_NEW, label='kite_model, wind at flying height')
    a3.plot(LDs, m7_6m, 'o-', color=C_ALT, label='kite_model, if published wind is at 6 m')
    a3.axhline(ref7, color=C_REF, ls='--', lw=1.2, label=f'published curve: {ref7:.0f} kW')
    a3.axvline(km.P['CLo'] / km.P['CDko'], color=C_GREY, ls=':', lw=1)
    a3.text(km.P['CLo'] / km.P['CDko'] + 0.05, 0.02, 'kite_model.P', fontsize=7.5, color=C_GREY,
            transform=a3.get_xaxis_transform())
    _style(a3, 'Which kite L/D matches at 7 m/s mean?', 'kite lift-to-drag ratio C_L/C_D [-]',
           'mean electrical power at 7 m/s [kW]')
    a3.set_ylim(0, 1.3 * max(ref7, m7_6m.max(), m7_zbar.max()))
    a3.legend(frameon=False, fontsize=7.5, loc='upper left')
    _note(fig, f'Inputs from the Falcon brochure: F_max 3.5 t, tether {FALCON["r_max"]:.0f} m, reel-out power at '
               f'rated 130 kW (electrical, used as generator limit), V9 kite 60-80 m² flat; 73 kg for 60 m² '
               f'(AWE course table), scaled with area. Assumed: reel-out starts at {FALCON["r_min"]:.0f} m. '
               f'Electrical = mechanical x {ETA:.2f} (reel-out) and / {ETA:.2f} (reel-in), Joshi et al. 2024. '
               'The brochure gives no reference height for its wind speed.')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / 'falcon.png', dpi=150)

    for mn, r, m1, m2 in zip(means, ref_mean, mod_mean, mod_6m):
        record('Kitepower Falcon (product)', f'mean power, Rayleigh {mn:.0f} m/s, wind at flying height',
               r, m1, 'kW', 'reference height of the published curve unknown')
        record('Kitepower Falcon (product)', f'mean power, Rayleigh {mn:.0f} m/s, if wind at 6 m',
               r, m2, 'kW', 'log profile z0 = 0.07 m (van der Vlugt 2019)')
    record('Kitepower Falcon (product)', 'kite L/D at which the model matches the mean power at 7 m/s',
           ref7, ref7, 'kW', f'wind at flying height: L/D = {LD_zbar:.2f}; wind at 6 m: L/D = {LD_6m:.2f} '
           f'(nan = not reached for L/D up to {LDs.max():.0f}; kite_model.P: {km.P["CLo"] / km.P["CDko"]:.1f})')
    record('Kitepower Falcon (product)', 'max. electrical cycle power', FALCON['curve_P'].max() / 1e3,
           curves[A_ref].max() / 1e3, 'kW', 'generator limit 130 kW electrical during reel-out')
    return dict(curves=curves, old=old, scale_6m=scale_6m, ref_mean=ref_mean, mod_mean=mod_mean, mod_6m=mod_6m,
                LD_zbar=LD_zbar, LD_6m=LD_6m)


# =====================================================================================
# 4. SKYSAILS PN-14
# =====================================================================================
def case_skysails():
    """Annual energy and rated power of the PN-14 against the model with the data sheet's inputs."""
    sk = SKYSAILS
    beta = km.P['beta_o']
    zmax = sk['r_max'] * sin(radians(beta))
    zmin = 2 * sk['z_flight'] - zmax                 # so that the average height is 200 m
    S = sk['area_flat'] * sk['proj_over_flat']
    d = sk['d_tether']
    rho_t = sk['tether_mass_per_m'] / (np.pi / 4 * d**2)
    # Tether force limit: the data sheet gives none. ASSUMED: the tether is sized like the
    # model sizes it (allowable stress kite_model.P['sigma']), so F_max follows from d.
    Fmax = km.P['sigma'] * np.pi / 4 * d**2
    p = dict(km.P, rho_t=rho_t, P_gen_max=sk['P_gen'])
    geo_update = dict(rho=sk['rho'])
    P_new = model_curve(V_GRID, S, sk['mass'], zmin, zmax, Fmax, p, geo_update=geo_update)
    P_old = model_curve(V_GRID, S, sk['mass'], zmin, zmax, Fmax, dict(p, **FIRST_VERSION),
                        geo_update=geo_update)

    means = np.arange(5.0, 12.01, 1.0)
    vv = V_GRID
    aep = lambda P: np.array([np.sum(rayleigh(mn, vv) * P) * 8760 / 1e6 for mn in means])  # MWh
    aep_new, aep_old = aep(P_new), aep(P_old)
    i9 = int(np.argmin(abs(means - 9)))

    # What kite lift-to-drag ratio would the data sheet imply? (kite drag varied, C_L fixed)
    LDs = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    aep9_LD = []
    for LD in LDs:
        P_LD = model_curve(V_GRID, S, sk['mass'], zmin, zmax, Fmax, dict(p, CDko=p['CLo'] / LD),
                           geo_update=geo_update)
        aep9_LD.append(np.sum(rayleigh(9.0, vv) * P_LD) * 8760 / 1e6)
    aep9_LD = np.array(aep9_LD)
    LD_match = float(np.interp(760.0, aep9_LD, LDs))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))
    a1.plot(V_GRID, P_new / 1e3, color=C_NEW, lw=2, label='kite_model (data-sheet inputs)')
    a1.plot(V_GRID, P_old / 1e3, color=C_OLD, lw=1.5, label='kite_model first version')
    a1.plot([sk['v_rated']], [sk['P_rated_cycle'] / 1e3], 'o', color=C_REF, ms=8,
            label='data sheet: rated 200 kW at 13 m/s')
    a1.axvline(sk['v_cutout'], color=C_REF, ls=':', lw=1)
    a1.text(sk['v_cutout'] - 0.3, 5, 'cut-out\n(data sheet)', ha='right', fontsize=7.5)
    _style(a1, 'SkySails PN-14: cycle power', 'wind speed at 200 m [m/s]', 'cycle power [kW]')
    a1.legend(frameon=False, fontsize=8, loc='upper left')
    a1.set_xlim(0, 26)

    a2.plot(means, aep_new, 'o-', color=C_NEW, label='kite_model')
    a2.plot(means, aep_old, 'o-', color=C_OLD, label='kite_model first version')
    a2.plot([9], [sk['AEP_9'] / 3.6e9], 's', color=C_REF, ms=9, label='data sheet: 760 MWh at 9 m/s')
    a2.text(9.15, sk['AEP_9'] / 3.6e9, f'model {aep_new[i9]:.0f} MWh\n({100 * (aep_new[i9] / 760 - 1):+.0f}%)',
            fontsize=8, va='top')
    _style(a2, 'Annual energy (Rayleigh, wind at 200 m)', 'annual mean wind speed at 200 m [m/s]',
           'AEP [MWh/yr]')
    a2.legend(frameon=False, fontsize=8, loc='upper left')

    a3.plot(LDs, aep9_LD, 'o-', color=C_NEW, label='kite_model, AEP at 9 m/s')
    a3.axhline(760, color=C_REF, ls='--', lw=1.2, label='data sheet: 760 MWh')
    a3.axvline(km.P['CLo'] / km.P['CDko'], color=C_GREY, ls=':', lw=1)
    a3.text(km.P['CLo'] / km.P['CDko'] + 0.05, aep9_LD.min(), 'kite_model.P\n(TU Delft V3 kite)',
            fontsize=7.5, color=C_GREY, va='bottom')
    a3.annotate(f'match at L/D = {LD_match:.1f}', (LD_match, 760), (LD_match + 0.15, 620),
                fontsize=8, arrowprops=dict(arrowstyle='->', color=C_REF))
    _style(a3, 'Which kite L/D matches the data sheet?', 'kite lift-to-drag ratio C_L/C_D [-]',
           'AEP at 9 m/s mean [MWh/yr]')
    a3.legend(frameon=False, fontsize=8, loc='upper left')
    _note(fig, f'Inputs from the PN-14 data sheet: kite 170 m² flat ({S:.0f} m² projected, ratio from the AWE course '
               f'table), {sk["mass"]:.0f} kg kite + pod, tether 14 mm, 0.245 kg/m, 800 m, air density 1.2, '
               f'generator 380 kW, 0% losses (as the data sheet). Assumed: F_max = {Fmax / 1e3:.0f} kN from the '
               f'tether diameter and the model\'s allowable stress; elevation {beta:.0f}°; average height 200 m.')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / 'skysails.png', dpi=150)

    record('SkySails PN-14 (product)', 'AEP at 9 m/s mean (Rayleigh, wind at 200 m)', 760.0,
           aep_new[i9], 'MWh/yr', 'same wind definition as the model')
    record('SkySails PN-14 (product)', 'kite L/D at which the model gives 760 MWh', 760.0, 760.0,
           'MWh/yr', f'L/D = {LD_match:.2f} (kite_model.P: {km.P["CLo"] / km.P["CDko"]:.1f})')
    record('SkySails PN-14 (product)', 'AEP at 9 m/s, first version', 760.0, aep_old[i9], 'MWh/yr')
    i13 = int(np.argmin(abs(V_GRID - 13)))
    record('SkySails PN-14 (product)', 'cycle power at rated wind 13 m/s', 200.0, P_new[i13] / 1e3, 'kW',
           'data sheet: "up to" 200 kW')
    return dict(P_new=P_new, P_old=P_old, means=means, aep_new=aep_new, aep_old=aep_old,
                LD_match=LD_match)


CASES = dict(three_regime=case_three_regime, vdv2019=case_vdv2019, falcon=case_falcon,
             skysails=case_skysails)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(CASES)
    for n in names:
        t0 = time.time()
        CASES[n]()
        print(f'{n}: done in {time.time() - t0:.0f} s', flush=True)
    df = pd.DataFrame(SUMMARY)
    path = OUT / ('summary.csv' if len(names) == len(CASES) else f'summary_{"_".join(names)}.csv')
    df.to_csv(path, index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 70)
    print(df[['reference', 'quantity', 'reference_value', 'model_value', 'difference_pct']].to_string(index=False))
