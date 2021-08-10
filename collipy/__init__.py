"""
collipy
=======

Provides an intuitive simple way to inject some particles

"""

from .particles import pdg
from .helper import fit, n_sigma, breit_wigner, expon
from .accelerator.accelerator import Accelerator
from .injection import InjectionCollection, DecayMode
from .helper import function_timer
