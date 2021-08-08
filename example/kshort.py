"""
K-short data collection and analysis
"""
import pickle
from pathlib import Path
import collipy as cp
import numpy as np


@cp.function_timer
def collect():
    alpha = 10
    particle = 'k-short'
    momentum = 3.7
    n = 200
    mode = cp.DecayMode(1, 2, 0)
    threshold = 0.15
    ac = cp.Accelerator(alpha)

    def cond(inj):
        """verify pi-plus and pi-minus (you will be surprised from some weird results)"""
        if inj.track.k.iloc[0] * inj.track.k.iloc[1] < 0:
            return True
        else:
            return False

    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    return data


def pt(k, sd_k):
    a, b = [1.94104212e-03, 2.16085896e-07]
    sd_a, sd_b = [6.58324160e-06, 2.53692179e-07]
    res = a / (k - b)
    sd_res = np.abs(res) * np.sqrt((sd_a/a)**2 + (sd_k/(k-b))**2 + (sd_b/(k-b))**2)
    return res, sd_res


def e(ph, sd_ph):
    a, b = [48.53818409, -20.31709717]
    sd_a, sd_b = [0.02454786, 0.55644483]
    res = (ph - b) / a
    sd_res = np.sqrt((sd_ph/a)**2 + (res*sd_a/a)**2 + (sd_b/a)**2)
    return res, sd_res


if __name__ == '__main__':
    path = Path('data/kshort.pickle')
    folder = 'data'
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()
