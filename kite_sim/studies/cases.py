"""
cases.py - Design cases, the cases each study needs, and the result cache
=========================================================================
"""

import hashlib
import json
import pickle
from pathlib import Path

from .. import model as km
from .. import wind as da
from .settings import (BASE, CACHE_FILE, FMAX_CLASSES, FMAX_VALUES, HEIGHT_CLASSES, HEIGHT_WINDOWS,
                       LD_RANGE, REPO, S_MAX, SENS_CLASS, SENSITIVITY, SITE_ELEVATION,
                       USE_SITE_ELEVATION)


# =====================================================================================
# CASES AND CACHE
# =====================================================================================
def make_case(cls, zmin=None, zmax=None, Fmax=None, P_target=None, **params):
    """
    One design case: wind class + design inputs + changes to the model parameters.

    Inputs
        cls                        : wind class number
        zmin, zmax, Fmax, P_target : [m], [m], [N], [W]; None takes the BASE value
        params                     : changes to kite_model.P, e.g. w=2.4

    Output
        dict that fully defines the case (also the cache key). It includes the class's
        site elevation z_site, so new elevation data recomputes the case.
    """
    return dict(cls=int(cls),
                z_site=float(SITE_ELEVATION.get(int(cls), 0.0)) if USE_SITE_ELEVATION else 0.0,
                zmin=float(zmin if zmin is not None else BASE['zmin']),
                zmax=float(zmax if zmax is not None else BASE['zmax']),
                Fmax=float(Fmax if Fmax is not None else BASE['Fmax']),
                P_target=float(P_target if P_target is not None else BASE['P_target']),
                params={k: float(v) for k, v in sorted(params.items())},
                S_max=S_MAX)


def _code_tag():
    """
    Fingerprint of the model code, data code and wind data file; the cache resets when any
    of them changes.

    Code: kite_model.py, data_analysis.py and every module in kite_sim/model/ and
    kite_sim/wind/. The data file is included because new data changes the classes without
    changing the case keys. Line endings are normalised, so CRLF (Windows) and LF
    (Linux/macOS) checkouts give the same fingerprint.
    """
    files = [REPO / 'kite_model.py', REPO / 'data_analysis.py']
    for package in ('model', 'wind'):
        files += sorted((REPO / 'kite_sim' / package).glob('*.py'), key=lambda f: f.name)
    h = hashlib.md5()
    for f in files + [Path(da.DATA_FILE)]:
        h.update(f.read_bytes().replace(b'\r\n', b'\n'))
    h.update(json.dumps(km.NUM, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _key(case):
    return json.dumps(case, sort_keys=True)


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as fh:
            cache = pickle.load(fh)
        if cache.get('tag') == _code_tag():
            return cache
    return dict(tag=_code_tag(), results={})


def save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as fh:
        pickle.dump(cache, fh)


# =====================================================================================
# STUDIES: the cases each figure needs
# =====================================================================================
N_CLASSES = len(da.CLASS_NAMES)


def cases_per_class():
    return [make_case(c) for c in range(1, N_CLASSES + 1)]


def cases_ld_range():
    """Every class at the baseline and at each kite L/D in LD_RANGE (C_L,o fixed)."""
    return cases_per_class() + [make_case(c, CDko=km.P['CLo'] / LD)
                                for LD in LD_RANGE for c in range(1, N_CLASSES + 1)]


def cases_height():
    return [make_case(c, zmin=a, zmax=b) for c in HEIGHT_CLASSES for a, b in HEIGHT_WINDOWS]


def cases_fmax():
    return [make_case(c, Fmax=F) for c in FMAX_CLASSES for F in FMAX_VALUES]


def cases_sensitivity():
    cases = [make_case(SENS_CLASS)]
    for key, lo, hi, _ in SENSITIVITY:
        cases += [make_case(SENS_CLASS, **{key: lo}), make_case(SENS_CLASS, **{key: hi})]
    return cases
