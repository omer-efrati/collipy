"""
Pi-0 data collection and analysis
"""

import pickle
from pathlib import Path
import collipy as cp
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


@cp.function_timer
def collect(cond):
    alpha = 1
    particle = 'pi-0'
    momentum = 2
    n = 20_000
    mode = cp.DecayMode(0, 0, 2)
    threshold = 0.05
    ac = cp.Accelerator(alpha)

    def cond(inj):
        """
        Condition for uncollimated ECAL indications for 2 GeV momentum

        Parameters
        ----------
        inj

        Returns
        -------
            bool
                True if two ECAL readings are at least 10cm apart
        """
        u = np.array([inj.vertex.x.iloc[0], inj.vertex.y.iloc[0], inj.vertex.z.iloc[0]])
        v = np.array([inj.vertex.x.iloc[1], inj.vertex.y.iloc[1], inj.vertex.z.iloc[1]])
        d = np.linalg.norm(u-v)
        if 10 <= d:
            return True
        return False

    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    return data


def e(ph, sd_ph):
    a, b = [48.53818409, -20.31709717]
    sd_a, sd_b = [0.02454786, 0.55644483]
    res = (ph - b) / a
    sd_res = np.sqrt((sd_ph/a)**2 + (res*sd_a/a)**2 + (sd_b/a)**2)
    return res.to_numpy(), sd_res.to_numpy()


def get_r_hat(vertex) -> (float, float):
    """Get distance from point of injection to vertex
    *** valid function for injections with a single vertex only"""
    u = np.array([vertex.x, vertex.y, vertex.z])
    du = np.array([vertex.dx, vertex.dy, vertex.dz])
    norm = np.linalg.norm(u)
    dnorm = np.linalg.norm(u * du) / norm
    hat = u / norm
    dhat = hat * np.sqrt((du / u)**2 + (dnorm/norm)**2)
    return hat, dhat


if __name__ == '__main__':
    path = Path('data/pizero.pickle')
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()

    _, _, ecal = data.get_df()
    mass, dmass = [], []
    for _, inj_ecal in ecal.groupby(level=0):
        p, dp = e(inj_ecal.ph, inj_ecal.dph)
        p1_norm, dp1_norm = p[0], dp[0]
        p2_norm, dp2_norm = p[1], dp[1]
        r1_hat, dr1_hat = get_r_hat(inj_ecal.iloc[0])
        r2_hat, dr2_hat = get_r_hat(inj_ecal.iloc[1])
        p1 = p1_norm * r1_hat
        dp1 = p1 * np.sqrt((dr1_hat/r1_hat)**2 + (dp1_norm/p1_norm)**2)
        p2 = p2_norm * r2_hat
        dp2 = p2 * np.sqrt((dr2_hat / r2_hat) ** 2 + (dp2_norm / p2_norm) ** 2)
        mass.append(p1 @ p2)
        dmass.append(np.sqrt(np.sum((p1 * dp2)**2 + (p2 * dp1)**2)))
