"""
plotting.py - Kite design studies: wind classes -> kite sizing -> figures
=========================================================================

WHAT IT DOES
    Links the wind classes (data_analysis.py) to the kite model (kite_model.py): for every
    case (wind class + design inputs + parameter changes) it sizes the kite for the target
    mean power and plots how the kite changes.

FIGURES (outputs/figures/)
    1 size_per_class       kite area and mass (kite + tether) per wind class
    2 size_vs_height       kite area, mass and tether against flying height
    3 downtime_per_class   hours per year out of operation and cut-in speed per class
    4 power_curves         power curve of the kite designed for each class
    5 fmax_effect          kite area and mass against the maximum tether force
    6 sensitivity          tornado chart: which model parameter moves the kite area most
    7 stability_cf         power variability (stability criterion), mean / max cycle power
    8 wind_classes         number of data points (areas) per mean wind speed, per class
    9 size_range_ld        kite area and downtime per class over the measured kite L/D range
   10 class_overview       kite area, airborne + tether mass and downtime per class (baseline)

HOW TO RUN
    python plotting.py                                all figures
    python plotting.py size_per_class sensitivity     only these figures
    python plotting.py --workers 4                    number of parallel processes
    python plotting.py --fresh                        ignore cached results and recompute

    Results are cached in outputs/kite_cache.pkl, keyed by the case and by the contents of
    the model and data code (kite_model.py, data_analysis.py, kite_sim/model/, kite_sim/wind/)
    and the wind data file. A changed study setting recomputes only the changed cases; changed
    model code recomputes everything.

RUN TIME
    One case (a full sizing) takes about 1-3 min on one core with kite_model.FAST; cases run in
    parallel. All figures need 96 cases (about 95 min on 2 cores, 30 min on 6). The committed
    cache (outputs/kite_cache.pkl) holds all of them for the current code and data, so the
    figures redraw in seconds.

SITE ELEVATION
    Each case uses the mean ground elevation of its wind class (data_analysis.class_elevations)
    for the air density. USE_SITE_ELEVATION = False (kite_sim/studies/settings.py) uses sea
    level everywhere; so does data without an elevation column.

CODE (kite_sim/studies/; re-exported here, so  import plotting as pl  works as before)
    settings.py           baseline design, study ranges, colours: edit these for other studies
    cases.py              make_case(), the cases each figure needs, result cache
    runner.py             run_case(), run_cases() (parallel)
    style.py              shared axis style and labels
    figures_per_class.py  figures 1, 3, 4, 7, 8, 10
    figures_studies.py    figures 2, 5, 6, 9
    cli.py                FIGURES and main()
"""

from kite_sim.studies import *      # noqa: F401,F403  list in kite_sim/studies/__init__.py
from kite_sim.studies import REPO, main

HERE = REPO                         # old name of the repository folder, kept for scripts that use it


if __name__ == '__main__':
    main()
