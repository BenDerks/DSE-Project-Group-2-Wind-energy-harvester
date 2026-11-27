"""
numerics.py - Numerical settings (FAST, ACCURATE, NUM) and the root finder solve()
==================================================================================

Step sizes and search grids are numerical choices, not physics: coarser settings run faster
but find the optimum less precisely.
    FAST      default, for parameter studies. About 7x faster than ACCURATE and ~1% less
              mean power for the example kite (so ~1.5% more kite area).
    ACCURATE  used for the worked example in the model doc; use it to check a final design.
NUM is the active setting: change the line  NUM = FAST  below to switch.
"""


# =====================================================================================
# NUMERICAL SETTINGS
# =====================================================================================
FAST = dict(
    n_fo=60,                          # reel-out factors tried per elevation (operating_point)
    fi_ratios=[0.03, 0.07, 0.15, 0.3],  # trial reel-in forces / F_max (bounds: Schelbergen 2020)
    beta_step=3.0,                    # [deg] elevation step when shedding force in strong wind
    dT=0.05,                          # reel-in time step as a fraction of dl / vw
    n_transition=15,                  # angle steps in the transition
    tol_state=1e-6,                   # convergence tolerance of the gravity iteration (on L/D)
    p_skip=1e-4,                      # sizing loop skips wind bins less probable than this
    S_tol=0.5,                        # [m2] tolerance on the kite area in the sizing loop
)
ACCURATE = dict(n_fo=150, fi_ratios=[0.03, 0.06, 0.1, 0.15, 0.25, 0.4], beta_step=1.0,
                dT=0.02, n_transition=50, tol_state=1e-8, p_skip=0.0, S_tol=0.2)
NUM = FAST


def solve(fun, a, b, xtol, maxiter=60):
    """
    Find x in [a, b] with fun(x) = 0 (Illinois variant of regula falsi; replaces scipy's brentq).

    Simple and robust, so the model needs only numpy. fun(a) and fun(b) must have opposite
    signs; stops when the bracket is smaller than xtol.
    """
    fa, fb = fun(a), fun(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError('solve(): no sign change between a and b')
    side = 0
    for _ in range(maxiter):
        x = (a * fb - b * fa) / (fb - fa)       # secant through the two bracket ends
        fx = fun(x)
        if fx * fb > 0:                         # root lies between a and x
            b, fb = x, fx
            if side == -1:
                fa /= 2                         # Illinois step: avoid a stuck end
            side = -1
        else:                                   # root lies between x and b
            a, fa = x, fx
            if side == +1:
                fb /= 2
            side = +1
        if abs(b - a) < xtol or fx == 0:
            break
    return x
