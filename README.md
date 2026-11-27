# DSE-Project-Group-2-Wind-energy-harvester
Project 2 - Open-hardware wind energy harvester for emergency and disaster relief

## Kite model code

| Run / import | What it is | Code |
|---|---|---|
| `kite_model.py` | Soft-kite pumping-cycle model; `python kite_model.py` runs a worked example | `kite_sim/model/` |
| `data_analysis.py` | Wind classes from `wind_results_2024.csv`; `python data_analysis.py` writes the class tables | `kite_sim/wind/` |
| `plotting.py` | Design studies and figures; `python plotting.py` (see its docstring for options) | `kite_sim/studies/` |
| `verify_power_curve.py`, `verification/` | Checks of the model against references | |

The three top-level files re-export everything from their package, so `import kite_model as km`
and the other imports work as before. The module list is in `kite_sim/__init__.py`.

Needs `numpy`, `pandas` and `matplotlib`. Install `numba` too (`pip install numba`): it compiles
the inner loops of the model, so a kite sizing takes seconds instead of minutes. Without numba
the code gives the same results, only slower.
