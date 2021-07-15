import numpy as np
from scipy.constants import hbar, eV, c, giga, centi

# Unit conversion constants
SEC = (hbar / (giga * eV)) ** -1              # 1 GeV^-1 = sec * 1 s
METER = ((hbar * c) / (giga * eV)) ** -1      # 1 GeV^-1 = meter * 1 m


class Particle:
    """Information about a particle from https://pdglive.lbl.gov
        * all attributes in units of GeV or seconds
        * all sizes of the form (float, float) stands for: (measurement, uncertainty)
    """
    def __init__(self, name: str, mass: (float, float), tau=(np.inf, 0), decay_modes={}):
        """
        Parameters
        ----------
        name
            name as stated at geant simulator menu
        mass
            mass in MeV units (as writen at PDG)
        tau
            life time in seconds units (as writen at PDG)
        decay_modes
            decay modes for this particle
            {1: (gamma_1/gamma, uncertainty1), 2: (gamma_2/gamma, uncertainty2) , ...}
        """
        self.name = name
        self.mass = (mass[0] * 10**-3, mass[1] * 10**-3)    # 10**-3 for MeV to GeV
        self.tau = tau
        self.decay_modes = decay_modes

    def __repr__(self):
        return self.name

    def get_p_best(self, lb: float, rb: float) -> float:
        """Calculate the most probable momentum in GeV units to decay at the interval [left, right]

        Parameters
        ----------
        lb
            left boundary for desired decay in cm units
        rb
            right boundary for desired decay in cm units
        Returns
        -------
        p
            momentum with highest chance to decay at the interval in GeV units
        """
        if not (0 <= lb < rb):
            raise ValueError('interval should sustain the inequality 0 <= interval[0] < interval[1]')
        if np.isinf(self.tau):
            raise ValueError('this particle will not decay at all')
        if lb == 0:
            return 0
        t = self.tau[0] * SEC
        m = self.mass[0]
        left = lb * centi * METER
        right = rb * centi * METER
        return (m * (right - left)) / (t * np.log(right / left))

    def get_p_chance(self, p: float, lb: float, rb: float) -> float:
        """Calculate the probability for this particle with momentum $p to decay within $interval

        Parameters
        ----------
        p
            momentum in GeV units
        lb
            left boundary for desired decay in cm units
        rb
            right boundary for desired decay in cm units
        Returns
        -------
        probability to decay
        """
        if not (0 <= lb < rb):
            raise ValueError('interval should sustain the inequality 0 <= interval[0] < interval[1]')
        if np.isinf(self.tau[0]):
            # particle will not decay
            return 0
        t = self.tau[0] * SEC
        m = self.mass[0]
        left = lb * centi * METER
        right = rb * centi * METER
        if p == 0:
            return 1
        return np.exp(-t ** -1 * (m * left) / p) - np.exp(-t ** -1 * (m * right) / p)


photon = Particle('photon', (0, 0))
electron = Particle('electron', (0.5109989461, 0.0000000031))
muon = Particle('muon', (105.6583745, 0.0000024))
k_s = Particle('k-short', (497.611, 0.013), (8.954e-11, 0.004e-11), {1: (30.69e-2, 0.05e-2), 2: (69.20e-2, 0.05e-2)})
pi_0 = Particle('pi-0', (134.9768, 0.0005), (8.43e-17, 0.13e-17),  {1: (98.823e-2, 0.034e-2), 2: (1.174e-2, 0.035e-2)})
pi_plus = Particle('pi-plus', (139.57039, 0.00018), (2.6033e-8, 0.0005e-8), {1: (99.98770e-2, 0.00004e-2)})
pi_minus = Particle('pi-minus', (139.57039, 0.00018), (2.6033e-8, 0.0005e-8), {1: (99.98770e-2, 0.00004e-2)})
