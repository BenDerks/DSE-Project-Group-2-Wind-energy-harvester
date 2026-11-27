"""
kite_sim.model - soft-kite pumping-cycle performance model (quasi-steady, with gravity)
=======================================================================================

Overview (what the model does, units, angle conventions, sources): see kite_model.py, which
re-exports everything listed here. One module per block of the model doc.
"""

from .params import P
from .numerics import FAST, ACCURATE, NUM, solve
from .geometry import geometry, coeffs
from .flight_state import state
from .phases import retraction, transition
from .operating_point import operating_point, _depowered, _with_transition_energy
from .sizing import weibull_bins, mean_power, size_kite, _refine_limits

__all__ = ['P', 'FAST', 'ACCURATE', 'NUM', 'solve', 'geometry', 'coeffs', 'state',
           'retraction', 'transition', 'operating_point', '_depowered', '_with_transition_energy',
           'weibull_bins', 'mean_power', 'size_kite', '_refine_limits']
