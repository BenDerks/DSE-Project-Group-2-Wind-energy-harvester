"""
kite_sim - soft-kite pumping-cycle model, wind classes and design studies
=========================================================================

Code behind the three top-level files, in modules of at most 300 lines. Each top-level file
re-exports its whole package, so use them as before:

    kite_model.py     ->  kite_sim/model/     the kite model (blocks A-H of the model doc)
    data_analysis.py  ->  kite_sim/wind/      wind classes from the wind-resource data
    plotting.py       ->  kite_sim/studies/   design studies, result cache and figures

    model/
        params.py             P: fixed parameters with their sources
        numerics.py           FAST, ACCURATE, NUM (numerical settings), solve() (root finder)
        geometry.py           A geometry(), B coeffs()
        flight_state.py       C state()
        phases.py             D retraction(), E transition()
        operating_point.py    F operating_point() and its helpers
        sizing.py             weibull_bins(), G mean_power(), sizing loop + H size_kite()
    wind/
        settings.py           data file, heights, class edges, wind-speed bins
        classes.py            load_data(), assign_classes(), class_summary()
        distributions.py      class_distributions(), class_elevations(), wind_bins(), ...
    studies/
        settings.py           baseline design, study ranges, colours
        cases.py              make_case(), the cases of each study, result cache
        runner.py             run_case(), run_cases() (parallel)
        style.py              shared axis style and labels
        figures_per_class.py  figures with one value per wind class
        figures_studies.py    height, F_max, sensitivity and L/D studies
        cli.py                FIGURES and main() (python plotting.py)
"""
