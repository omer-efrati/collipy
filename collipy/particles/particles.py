"""
Particles data from [1] PDG

Notes
-----
The data copied from [1] is stored in data/particles.csv file

References
----------
[1] Zyla, P. A., & others. (2020). Review of Particle Physics. PTEP, 2020(8), 083C01.
    https://doi.org/10.1093/ptep/ptaa104

"""

import os
import numpy as np
import pandas as pd
from scipy.constants import hbar, eV, c, giga, centi


class Particle:
    """
    Particle information and utilities

    Parameters
    ----------
    name : str
        String name of the particle as stated in the README.md file
    tau : tuple[float, float]
        Life time in second unit
    mass : tuple[float, float]
        Mass in MeV unit

    Attributes
    ----------
    name: str
        String name of the particle as stated in the README.md file
    mass: tuple[float, float]
        mass in GeV unit
    tau: tuple[float, float]
        Life time in second unit


    Notes
    -----
    * all attributes in units of GeV or seconds

    * all sizes of the form tuple[float, float] stands for: (value, uncertainty)

    * SEC is the factor to convert time units from [s] to [GeV^-1] i.e. SEC * t [s] = t [GeV^-1]

    * METER factor to convert length units from [m] to [GeV^-1] i.e. METER * x [m] = x [GeV^-1]

    """

    SEC = (hbar / (giga * eV)) ** -1
    METER = ((hbar * c) / (giga * eV)) ** -1

    def __init__(self, name: str, mass: tuple[float, float], tau: tuple[float, float]):
        self.name = name
        # converting units to GeV
        self.mass = mass[0] * 10**-3, mass[1] * 10**-3
        self.tau = tau

    def __repr__(self):
        return self.name

    def get_momentum_in_range(self, lb: float, rb: float) -> float:
        """
        Calculate the most probable momentum for `self` particle to decay within the interval [`lb`, `rb`]

        Parameters
        ----------
        lb
            left bound for desired decay in cm unit
        rb
            right bound for desired decay in cm unit
        Returns
        -------
        momentum
            momentum in GeV units

        """
        if not (0 <= lb < rb):
            raise ValueError("Boundaries should sustain the inequality 0 <= lb < rb")
        if np.isinf(self.tau[0]):
            raise ValueError("This particle will not decay at all")
        if lb == 0:
            return 0
        t = self.tau[0] * Particle.SEC
        m = self.mass[0]
        left = lb * centi * Particle.METER
        right = rb * centi * Particle.METER
        return (m * (right - left)) / (t * np.log(right / left))

    def get_momentum_probability(self, momentum: float, lb: float, rb: float) -> float:
        """
        Calculate the probability for `self` particle to decay within the interval [`lb`, `rb`]
        with momentum `momentum`

        Parameters
        ----------
        momentum
            momentum in GeV units
        lb
            left boundary for desired decay in cm unit
        rb
            right boundary for desired decay in cm unit
        Returns
        -------
        probability to decay

        """
        if not (0 <= lb < rb):
            raise ValueError("Boundaries should sustain the inequality 0 <= lb < rb")
        if np.isinf(self.tau[0]):
            # particle will not decay in any circumstances
            return 0
        t = self.tau[0] * Particle.SEC
        m = self.mass[0]
        left = lb * centi * Particle.METER
        right = rb * centi * Particle.METER
        if momentum == 0:
            return 1
        return np.exp(-t ** -1 * (m * left) / momentum) - np.exp(-t ** -1 * (m * right) / momentum)


def create_particles_dic():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'data', 'particles.csv'))
    dic = dict()
    for _, row in df.iterrows():
        dic[row['name']] = Particle(row['name'],
                                    (row['mass[MeV]'], row['sd_mass[MeV]']),
                                    (row['tau[s]'], row['sd_tau[s]']))
    return dic


pdg = create_particles_dic()
