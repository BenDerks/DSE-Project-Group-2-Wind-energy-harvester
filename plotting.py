"""
plotting.py - Kite design studies: wind classes -> kite sizing -> figures
=========================================================================

WHAT IT DOES
    Links the wind classes (data_analysis.py) to the kite model (kite_model.py): for every
    case (wind class + design inputs + parameter changes) it sizes the kite for the target
    mean power and plots how the kite changes.

FIGURES (outputs/results_min_area/ or outputs/results_multi_objective/, see --objective)
    1 size_vs_height       kite area against z_min and z_max, every class (z_max <= 300 m)
    2 size_vs_ld           kite area against the kite lift-to-drag ratio, one line per class
    3 size_vs_fmax         kite area against the maximum tether force, one line per class
    4 power_curves         power curve of the kite designed for each class (extra)
    5 sensitivity          tornado chart: which model parameter moves the kite area most (extra)
    6 wind_classes         number of data points (areas) per mean wind speed, per class (extra)
    7 best_designs         FINAL SUMMARY: best design per class, figure + best_designs.csv
    8 tradeoff             (multi only) kite area against longest outage per candidate, Pareto
                           front and chosen design; every candidate in all_designs.csv

HOW TO RUN
    python plotting.py                                all figures, best design = smallest kite
    python plotting.py --objective multi              best design = weighted score of kite area,
                                                      downtime and longest continuous downtime over
                                                      every height window x F_max (needs the hourly
                                                      ERA5 wind of windres and pyarrow)
    python plotting.py size_vs_ld size_vs_fmax        only these figures
    python plotting.py --workers 4                    number of parallel processes
    python plotting.py --fresh                        ignore cached results and recompute

    Results are cached in outputs/kite_cache.pkl, keyed by the case and by the contents of
    the model and data code (kite_model.py, data_analysis.py, kite_sim/model/, kite_sim/wind/)
    and the wind data file. A changed study setting recomputes only the changed cases; changed
    model code recomputes everything.

RUN TIME
    One case (a full sizing) takes about 1-3 min on one core with kite_model.FAST; cases run in
    parallel. All figures need about 165 cases (about 45 min on 8 cores, 3 h on 2). Cached
    cases are not recomputed; a model or data change resets the cache.

SITE ELEVATION
    Each case uses the mean ground elevation of its wind class (data_analysis.class_elevations)
    for the air density. USE_SITE_ELEVATION = False (kite_sim/studies/settings.py) uses sea
    level everywhere; so does data without an elevation column.

CODE (kite_sim/studies/; re-exported here, so  import plotting as pl  works as before)
    settings.py           baseline design, study ranges, colours: edit these for other studies
    cases.py              make_case(), the cases each figure needs, result cache
    runner.py             run_case(), run_cases() (parallel)
    style.py              shared axis style and labels
    figures_per_class.py  figures 4, 6
    figures_studies.py    figures 1, 2, 3, 5
    selection.py          candidates and best design per objective (settings: OBJECTIVE_WEIGHTS)
    downtime.py           downtime and continuous downtime from the hourly ERA5 wind
    figures_summary.py    figures 7, 8
    cli.py                FIGURES and main()
"""

from kite_sim.studies import *      # noqa: F401,F403  list in kite_sim/studies/__init__.py
from kite_sim.studies import REPO, main

HERE = REPO                         # old name of the repository folder, kept for scripts that use it


if __name__ == '__main__':
    main()
