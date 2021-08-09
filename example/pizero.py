"""
pi-0 data collection and analysis
"""
import pickle
from pathlib import Path
import collipy as cp
import numpy as np


@cp.function_timer
def collect():
    alpha = 1
    particle = 'pi-0'
    momentum = 1
    n = 6*60*60
    mode = cp.DecayMode(0, 0, 2)
    threshold = 0.05
    ac = cp.Accelerator(alpha)
    data = ac.collect(particle, momentum, n, mode, threshold)
    return data

data = collect()
path = Path('data/pizero.pickle')
with open(path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
