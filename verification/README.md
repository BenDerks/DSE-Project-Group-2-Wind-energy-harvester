# Verification of the kite model

This folder checks `kite_model.py` against independent references, each run with the
reference's own inputs. Every value used is listed with its source in `references.py`,
`vdv2019_cases.py` and `three_regime.py`. Assumed values are marked ASSUMED in the code.

| File | What it does |
|---|---|
| `run_verification.py` | Runs all four cases, saves the figures and `summary.csv` in `outputs/verification/` |
| `three_regime.py` | Independent reference code: massless three-regime model (Luchsinger 2013), ported from R. Schmehl's `awe_on_mars/powercurve.py` (MIT) |
| `vdv2019_cases.py` | The two measured pumping cycles of van der Vlugt et al. (2019), re-run with `kite_model` |
| `references.py` | Product data: Kitepower Falcon (brochure, power curve), SkySails PN-14 (data sheet), drivetrain efficiency (Joshi et al. 2024) |
| `design_impact.py` | Kite size per wind class with the first model version and the current one |

## How to run

```
python verification/run_verification.py            # all cases, about 15 minutes
python verification/run_verification.py falcon     # one case: three_regime, vdv2019, falcon, skysails
python verification/design_impact.py               # uses the plotting.py cache
```

## Figures (in `outputs/verification/`)

| Figure | Reference | What to look at |
|---|---|---|
| `three_regime.png` | Independent code | Force laws agree exactly; the cycle differs because the reference reels in at a steady high elevation without climbing or a transition |
| `vdv2019.png` | Measured cycles and published model | Phase powers of the measurement, the published model and `kite_model` |
| `skysails.png` | SkySails PN-14 data sheet | Power curve, AEP and the kite L/D at which the model matches the data sheet |
| `falcon.png` | Kitepower Falcon brochure | Power curve and mean power, for two readings of the (unstated) wind height, and the L/D that would match |
| `design_impact.png` | - | What the model changes do to the kite area per wind class |

The overview of results, changes and reasons for remaining differences is in the
"Kite model verification" doc.
