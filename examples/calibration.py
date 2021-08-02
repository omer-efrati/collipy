"""
Calibration data collection and analysis
"""
import pickle
from pathlib import Path
import collipy as cp
import numpy as np
import pandas as pd
from collipy.helper import function_timer


@function_timer
def collect():
    alpha = 1
    particles = ['electron', 'muon', 'photon']
    momenta = np.arange(10, 110, 10)
    modes = [cp.DecayMode(0, 1, 1), cp.DecayMode(0, 1, 1), cp.DecayMode(0, 0, 1)]
    n = 200
    threshold = 0.05
    ac = cp.Accelerator(alpha)
    return ac.acollect(particles, modes, momenta, n, threshold)


if __name__ == '__main__':
    # path to data file
    path = Path('../data/calibration.pickle')
    folder = 'data'
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()
