"""
falcon_max_power.py - What limits the maximum power of the Falcon-size kite in kite_model?
=========================================================================================

Left:  published Falcon power curve against kite_model with the brochure inputs and with a
       higher tether force limit F_max (and without the generator limit).
Right: maximum electrical cycle power when one input at a time is changed, starting from
       the brochure inputs with F_max raised to 60 kN.

Inputs and assumptions as in run_verification.case_falcon (100 m2 flat kite, tether
200-350 m, electrical = mechanical x 0.86 for reel-out and / 0.86 for reel-in).

HOW TO RUN
    python verification/falcon_max_power.py        (about 10 minutes)
OUTPUT
    outputs/verification/falcon_max_power.png, falcon_max_power.csv
"""

import sys
from math import radians, sin
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
import run_verification as rv
from references import FALCON, ETA, falcon_curve

OUT = HERE.parent / 'outputs' / 'verification'
V = np.arange(1.0, 25.01, 0.5)          # power curves
V_MAX = np.arange(8.0, 25.01, 1.0)      # only to find the maximum

beta = km.P['beta_o']
S, m = rv._falcon_variants()[100.0]
P_GEN = FALCON['P_reel_out_el'] / ETA  # mechanical reel-out limit from the brochure's 130 kW


def curve(vs, F=FALCON['F_max'], gen=P_GEN, r_min=FALCON['r_min'], S_=S, m_=m, **params):
    zmin, zmax = r_min * sin(radians(beta)), FALCON['r_max'] * sin(radians(beta))
    p = dict(km.P, P_gen_max=gen, **params)
    return rv.model_curve(vs, S_, m_, zmin, zmax, F, p, eta=ETA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Left: power curves
    lines = [
        ('kite_model, brochure inputs (F_max 34 kN, generator 130 kW)', dict(), '#9ecae1', '-'),
        ('kite_model, F_max 60 kN, generator 130 kW', dict(F=60e3), '#2b6cb0', '-'),
        ('kite_model, F_max 100 kN, generator 130 kW', dict(F=100e3), '#08306b', '-'),
        ('kite_model, F_max 55 kN, no generator limit', dict(F=55e3, gen=None), '#d97706', '--'),
    ]
    curves = [(lab, curve(V, **kw), c, ls) for lab, kw, c, ls in lines]

    # Right: maximum power, one input changed at a time from the 60 kN case
    base = dict(F=60e3)
    variants = [
        ('reference: F_max 60 kN', {}),
        ('F_max 34 kN (brochure)', dict(F=FALCON['F_max'])),
        ('F_max 100 kN', dict(F=100e3)),
        ('generator 200 kW instead of 130 kW (electrical)', dict(gen=200e3 / ETA)),
        ('no generator limit', dict(gen=None)),
        ('reel-in L/D 3.5 instead of 2.0 (van der Vlugt 2019)', dict(CDki=km.P['CLi'] / 3.5)),
        ('max reeling speed 15 instead of 10 m/s', dict(vmax=15.0)),
        ('kite L/D 5 instead of 4', dict(CDko=km.P['CLo'] / 5.0)),
        ('reel-out from 150 m instead of 200 m (longer stroke)', dict(r_min=150.0)),
        ('kite 80 m² flat instead of 100', dict(S_=S * 0.8, m_=m * 0.8)),
    ]
    rows = []
    for lab, kw in variants:
        P = curve(V_MAX, **dict(base, **kw))
        rows.append(dict(change=lab, max_kW=P.max() / 1e3, at_wind=V_MAX[P.argmax()]))
        print(f'{lab:55s} {P.max() / 1e3:6.1f} kW at {V_MAX[P.argmax()]:.0f} m/s', flush=True)
    table = pd.DataFrame(rows)
    table.round(1).to_csv(OUT / 'falcon_max_power.csv', index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    v = np.linspace(0, 25, 251)
    a1.plot(v, falcon_curve(v) / 1e3, color='#3a3a3a', lw=2.6, label='Falcon, published (100 m², electrical)')
    for lab, P, c, ls in curves:
        a1.plot(V, P / 1e3, color=c, lw=1.8, ls=ls, label=lab)
    a1.axhline(0, color='#555555', lw=0.8)
    a1.set_xlim(0, 25)
    rv._style(a1, 'Falcon power curve against kite_model', 'wind speed [m/s] (model: at the average flying height)',
              'electrical cycle power [kW]')
    a1.legend(frameon=False, fontsize=7.5, loc='upper left')

    ref = table['max_kW'].iloc[0]
    order = table.iloc[1:].sort_values('max_kW')
    y = np.arange(len(order))
    a2.barh(y, order['max_kW'], color=['#2b6cb0' if v_ >= ref else '#9ecae1' for v_ in order['max_kW']])
    a2.axvline(ref, color='#3a3a3a', lw=1.2, ls='--')
    a2.axvline(FALCON['curve_P'].max() / 1e3, color='#d97706', lw=1.2, ls=':')
    a2.text(ref, len(order) - 0.4, f' reference {ref:.0f} kW', fontsize=7.5, va='bottom')
    a2.text(FALCON['curve_P'].max() / 1e3, -0.9, ' published\n plateau', fontsize=7.5, color='#d97706', va='top')
    for yi, val in zip(y, order['max_kW']):
        a2.text(val + 2, yi, f'{val:.0f} kW ({100 * (val / ref - 1):+.0f}%)', va='center', fontsize=7.5)
    a2.set_yticks(y, order['change'], fontsize=8)
    a2.set_xlim(0, 1.35 * order['max_kW'].max())
    a2.set_ylim(-1.3, len(order))
    rv._style(a2, 'Maximum power: one input changed at a time', 'maximum electrical cycle power [kW]')
    fig.text(0.01, 0.005, 'Falcon inputs from the brochure (100 m² flat, tether 350 m, reel-out from 200 m assumed, '
             'generator limit = 130 kW electrical during reel-out). Electrical = mechanical x 0.86 (reel-out), '
             '/ 0.86 (reel-in), Joshi et al. 2024. Right: maximum over 8-25 m/s.', fontsize=7.5, color='#9a9a9a')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / 'falcon_max_power.png', dpi=150)
    print('saved', OUT / 'falcon_max_power.png')


if __name__ == '__main__':
    main()
