"""
run_verification.py - Verify kite_model.py against independent code, test data and products
===========================================================================================

Runs kite_model.py with each reference's OWN inputs; one figure per reference plus a summary
table. Each case also runs the first version of the model (no transition energy, depower or
elevation optimisation) to show what the verification changes did.

    1 three_regime   independent reference CODE: massless three-regime model of
                     Luchsinger (2013), implemented by R. Schmehl (three_regime.py)
    2 vdv2019        published MODEL RESULTS and MEASUREMENTS of two pumping cycles of the
                     TU Delft 20 kW system (van der Vlugt et al. 2019; vdv2019_cases.py)
    3 falcon         PRODUCT: Kitepower Falcon 100 kW power curve (electrical)
    4 skysails       PRODUCT: SkySails PN-14 annual energy and rated power (wind at 200 m)

HOW TO RUN
    python verification/run_verification.py            all cases (about 10 minutes)
    python verification/run_verification.py falcon     one case

CODE
    common.py           settings, summary table, helpers (model_curve(), rayleigh(), record())
    cases_models.py     cases 1 and 2
    cases_products.py   cases 3 and 4

OUTPUT (outputs/verification/)
    *.png         figures
    summary.csv   one row per compared quantity: reference, model, difference
"""

import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import OUT, SUMMARY                                              # noqa: E402
from cases_models import case_three_regime, case_vdv2019                     # noqa: E402
from cases_products import case_falcon, case_skysails                        # noqa: E402


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
