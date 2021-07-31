"""
collipy
=======

Provides an intuitive simple way to inject some particles

TODO: add some discription

"""

from .particles import pdg
from .helper import fit, n_sigma
from .accelerator.acc import Accelerator
from .injection import InjectionCollection, DecayMode
