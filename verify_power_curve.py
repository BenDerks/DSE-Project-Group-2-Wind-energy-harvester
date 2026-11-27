"""
NOTE (27 Nov 2026): superseded by verification/run_verification.py (case "falcon"), which uses
the Falcon brochure inputs (kite size, F_max, generator limit) and electrical conversion. This
file is kept for reference only.

verify_power_curve.py - Compare the kite model with a published power curve
===========================================================================

WHAT THIS FILE DOES
    Checks kite_model.py against the power curve of an existing pumping kite system
    (100 m2 kite, 350 m tether). It computes the model's power curve for the same kite and
    tether, and the mean power both curves give for three mean wind speeds.

FIGURES (saved in outputs/figures/)
    verification_power_curve   reference curve against the model curve, for several F_max
    verification_mean_wind     energy per wind speed and mean power, at three mean wind speeds

WHAT TO LOOK FOR
    - Cut-in: where both curves start to give positive power.
    - Slope: how fast the power rises between cut-in and rated (kite size and L/D).
    - Plateau: the rated power (set by F_max * v_max in the model).
    - Mean power: the number that sizes our kite. Agreement here matters most.

DIFFERENCES THAT ARE EXPECTED (not errors in the model)
    - The reference is probably electrical power after the generator; the model gives
      mechanical power at the drum (model boundary). Drivetrain efficiency is ~0.8-0.9, so
      the model may be 10-20% higher in the plateau.
    - The reference goes below zero at low wind (energy used while not producing); the model
      gives 0 below cut-in. The mean power is shown with and without these negative values.
    - The wind speed on the reference x-axis may be at a different height than the model's
      (model: average flying height z_bar). Not stated on the figure.
    - The tether force limit, reel-out stroke and reeling speeds of the reference system are not
      given; F_max is therefore varied, and the stroke is an assumption (REEL_OUT_START).

HOW TO RUN
    python verify_power_curve.py            (takes a few minutes, one core)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import kite_model as km

HERE = Path(__file__).parent
FIG_DIR = HERE / 'outputs' / 'figures'

# =====================================================================================
# SETTINGS
# =====================================================================================
# Reference power curve, read off the published figure by eye (accuracy ~ +-3 kW).
# Wind speed [m/s], power [kW]
REF_V = np.array([0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 13, 14, 15,
                  17, 20, 25], dtype=float)
REF_P = np.array([-6.5, -5.5, -4, -2, 1, 5, 14, 21, 32, 50, 66, 79, 90, 97, 103, 110, 114,
                  116.5, 118, 119, 120.5, 122, 123.5])

# Reference system: 100 m2 kite, 350 m tether.
# The kite area of a commercial kite is normally quoted as the flat area; the model uses the
# projected area S = area_ratio * A. Both readings are computed (flat is the main one).
REF_AREA = 100.0                     # [m2]
TETHER_LENGTH = 350.0                # [m]  taken as the tether length at the end of reel-out
REEL_OUT_START = 200.0               # [m]  tether length at the start of reel-out.
                                     #      ASSUMPTION: not given for the reference system.

# Tether force limits to try [N] (not given for the reference system)
FMAX_VALUES = [40e3, 60e3, 80e3]
FMAX_MAIN = 60e3                     # used for the projected-area variant and figure 2

# Mean wind speeds for figure 2 [m/s], Rayleigh distribution (Weibull k = 2), the
# distribution used for annual energy estimates in IEC 61400-12-1 (power performance)
MEAN_WINDS = [5.0, 7.0, 9.0]
K_RAYLEIGH = 2.0

# Wind speeds at which the model power curve is computed [m/s]
V_GRID = np.arange(0.5, 25.01, 0.5)
DV = V_GRID[1] - V_GRID[0]

C_REF = '#3a3a3a'
C_LINES = ['#9ecae1', '#2b6cb0', '#08306b']
C_PROJ = '#d97706'
C_GREY = '#9a9a9a'


# =====================================================================================
# MODEL POWER CURVE
# =====================================================================================
def model_curve(S, Fmax, p=km.P):
    """
    Cycle power of the model at every wind speed in V_GRID.

    Inputs: S [m2] projected area, Fmax [N], p parameter dict.
    Output: array of cycle power [kW]; 0 below cut-in and above cut-out (as in mean_power).
    """
    beta = np.radians(p['beta_o'])
    zmin, zmax = REEL_OUT_START * np.sin(beta), TETHER_LENGTH * np.sin(beta)   # M1 inverted
    geo = km.geometry(zmin, zmax, Fmax, p)
    out = []
    for v in V_GRID:
        r = km.operating_point(v, S, geo, Fmax, p)
        out.append(r['Pc'] / 1e3 if isinstance(r, dict) else 0.0)
    return np.array(out)


def ref_curve(v, clip=False):
    """Reference power [kW] at wind speeds v (linear between the read-off points)."""
    P = np.interp(v, REF_V, REF_P)
    return np.maximum(P, 0.0) if clip else P


def rayleigh(mean):
    """Bin probabilities of a Rayleigh distribution on V_GRID (0.25-25.25 m/s)."""
    return km.weibull_bins(mean, K_RAYLEIGH, V_GRID, DV)


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


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_power_curve(curves):
    """Reference curve against the model curves."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    v_ref = np.linspace(0, 25, 251)
    ax.plot(v_ref, ref_curve(v_ref), color=C_REF, lw=2.5, label='reference (published)')
    for (F, P), col in zip(curves['flat'].items(), C_LINES):
        ax.plot(V_GRID, P, color=col, lw=1.8,
                label=f'model, flat area {REF_AREA:.0f} m², F_max {F / 1e3:.0f} kN')
    ax.plot(V_GRID, curves['proj'], color=C_PROJ, lw=1.8, ls='--',
            label=f'model, projected area {REF_AREA:.0f} m², F_max {FMAX_MAIN / 1e3:.0f} kN')
    ax.axhline(0, color='#555555', lw=0.8)
    _style(ax, 'Power curve: reference system against the kite model',
           'wind speed [m/s] (model: at the average flying height)', 'power [kW]')
    ax.set_xlim(0, 25)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.text(0.01, 0.01, f'Tether {REEL_OUT_START:.0f}-{TETHER_LENGTH:.0f} m during reel-out '
             f'(start assumed), elevation {km.P["beta_o"]:.0f}°. Model: mechanical power at the '
             f'drum; reference: probably electrical. Reference read off the figure (±3 kW).',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_mean_wind(curves):
    """Energy per wind speed (top) and mean power (bottom) for the three mean wind speeds."""
    P_model = curves['flat'][FMAX_MAIN]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, len(MEAN_WINDS), height_ratios=[1, 1.1])

    rows = []
    for i, vm in enumerate(MEAN_WINDS):
        pj = rayleigh(vm)
        ax = fig.add_subplot(gs[0, i])
        # Energy per wind speed: probability x power x hours, per m/s of wind speed
        e_ref = 8760 * pj * ref_curve(V_GRID) / 1e3 / DV          # [MWh/yr per m/s]
        e_mod = 8760 * pj * P_model / 1e3 / DV
        ax.bar(V_GRID, pj / DV * 100, DV * 0.9, color='#eeeeee', zorder=0)
        ax2 = ax.twinx()
        ax2.plot(V_GRID, e_ref, color=C_REF, lw=2, label='reference')
        ax2.plot(V_GRID, e_mod, color=C_LINES[1], lw=2, label='model')
        ax.set_ylabel('probability [% per m/s]', color=C_GREY)
        ax2.set_ylabel('energy [MWh/yr per m/s]')
        ax.set_title(f'Mean wind {vm:.0f} m/s', loc='left', fontsize=11, fontweight='bold')
        ax.set_xlabel('wind speed [m/s]')
        ax.set_xlim(0, 25)
        ax2.spines['top'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i == 0:
            ax2.legend(frameon=False, fontsize=8, loc='upper right')

        row = dict(ref=float(np.sum(pj * ref_curve(V_GRID))),
                   ref_pos=float(np.sum(pj * ref_curve(V_GRID, clip=True))),
                   proj=float(np.sum(pj * curves['proj'])))
        row.update({F: float(np.sum(pj * P)) for F, P in curves['flat'].items()})
        rows.append(row)

    # Mean power per mean wind speed
    ax = fig.add_subplot(gs[1, :])
    labels = [('ref', 'reference (published)', C_REF),
              ('ref_pos', 'reference, negative part set to 0', '#8c8c8c')]
    labels += [(F, f'model, flat {REF_AREA:.0f} m², F_max {F / 1e3:.0f} kN', c)
               for F, c in zip(FMAX_VALUES, C_LINES)]
    labels += [('proj', f'model, projected {REF_AREA:.0f} m², F_max {FMAX_MAIN / 1e3:.0f} kN', C_PROJ)]
    n = len(labels)
    width = 0.8 / n
    x = np.arange(len(MEAN_WINDS))
    for j, (key, lab, col) in enumerate(labels):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (j - (n - 1) / 2) * width, vals, width, color=col, label=lab)
        for b, val, r in zip(bars, vals, rows):
            txt = f'{val:.0f}' if key in ('ref', 'ref_pos') else \
                f'{val:.0f}\n{100 * (val / r["ref"] - 1):+.0f}%'
            ax.text(b.get_x() + b.get_width() / 2, val + 0.8, txt, ha='center', va='bottom',
                    fontsize=7)
    ax.set_xticks(x, [f'mean wind {vm:.0f} m/s' for vm in MEAN_WINDS])
    _style(ax, 'Mean power (percentages: difference with the published reference)',
           ylabel='mean power [kW]')
    ax.set_ylim(0, 1.25 * max(max(r.values()) for r in rows))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
    fig.text(0.01, 0.005, f'Rayleigh wind distribution (Weibull k = {K_RAYLEIGH:.0f}, IEC 61400-12-1). '
             f'Top row: model with flat area {REF_AREA:.0f} m² and F_max {FMAX_MAIN / 1e3:.0f} kN. '
             'Model has no drivetrain losses.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig, rows


# =====================================================================================
# MAIN
# =====================================================================================
if __name__ == '__main__':
    S_flat = km.P['area_ratio'] * REF_AREA          # flat 100 m2 -> projected S (M8)
    curves = dict(flat={}, proj=None)
    for F in FMAX_VALUES:
        print(f'model curve, flat area {REF_AREA:.0f} m2 (S = {S_flat:.0f} m2), F_max {F / 1e3:.0f} kN')
        curves['flat'][F] = model_curve(S_flat, F)
    print(f'model curve, projected area {REF_AREA:.0f} m2, F_max {FMAX_MAIN / 1e3:.0f} kN')
    curves['proj'] = model_curve(REF_AREA, FMAX_MAIN)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig_power_curve(curves)
    fig.savefig(FIG_DIR / 'verification_power_curve.png', dpi=150)
    fig, rows = fig_mean_wind(curves)
    fig.savefig(FIG_DIR / 'verification_mean_wind.png', dpi=150)

    print('\nMean power [kW]')
    for vm, r in zip(MEAN_WINDS, rows):
        print(f'  {vm:.0f} m/s: reference {r["ref"]:.1f} (production only {r["ref_pos"]:.1f}), '
              + ', '.join(f'model {F / 1e3:.0f} kN {r[F]:.1f}' for F in FMAX_VALUES)
              + f', projected {r["proj"]:.1f}')
    print(f'saved figures in {FIG_DIR}')
