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
    momentum = 1.2
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
        u = np.array([inj.ecal.x.iloc[0], inj.ecal.y.iloc[0], inj.ecal.z.iloc[0]])
        v = np.array([inj.ecal.x.iloc[1], inj.ecal.y.iloc[1], inj.ecal.z.iloc[1]])
        d = np.linalg.norm(u-v)
        if 10 <= d:
            return True
        return False

    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    return data


def e(ph, sd_ph):
    a, b = [0.02060234, 0.41857967]
    sd_a, sd_b = [1.04194919e-05, 1.13030934e-02]
    res = a * ph + b
    sd_res = np.sqrt((sd_ph*a)**2 + (sd_a*ph)**2 + sd_b**2)
    return np.array(res), np.array(sd_res)


def get_r_hat(ecal) -> (float, float):
    """Get distance from point of injection to ecal
    *** valid function for injections with a single ecal only"""
    u = np.array([ecal.x, ecal.y, ecal.z])
    du = np.array([ecal.dx, ecal.dy, ecal.dz])
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
    m, dm = [], []
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
        dot = p1 @ p2
        ddot = np.sqrt(np.sum((p1 * dp2)**2 + (p2 * dp1)**2))
        mass = 2 * (p1_norm*p2_norm - dot)
        dmass = 2 * np.sqrt((p2_norm*dp1_norm)**2 + (p1_norm*dp2_norm)**2 + ddot**2)
        m.append(np.sqrt(mass))
        dm.append(dmass/(2 * np.sqrt(mass)))
    m = np.array(m)
    dm = np.array(m)

    # dm = dm[0.1 < m]
    # m = m[0.1 < m]
    # z = stats.zscore(m)
    # m = m[z < 3]
    # dm = dm[z < 3]
    y, x = np.histogram(a=m, bins='auto')
    dy = np.sqrt(y)
    dx = ((x[1] - x[0]) / 2) * np.ones_like(y)
    x = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    x, dx, y = x[0 < dy], dx[0 < dy], y[0 < dy]
    dy = dy[0 < dy]
    beta_initial = [0, 9e-4, cp.pdg['pi-0'].mass[0], 0.0076]
    out, chisq_red, p_value = cp.fit(x, y, dy, cp.breit_wigner, beta_initial)

    _, mass_fit = plt.subplots(subplot_kw=dict(title=f'$\pi^0$ Mass Distribution',
                                               ylabel=r'Events', xlabel=r'$Mass[GeV]$'))
    _, mass_res = plt.subplots(subplot_kw=dict(title=f'$\pi^0$ Mass Distribution Residuals Plot',
                                               ylabel=r'Residuals', xlabel=r'$Mass[GeV]$'))
    mass_fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
    mass_fit.plot(np.linspace(0.344, 0.65, 10_000), cp.breit_wigner(out.beta, np.linspace(0.344, 0.65, 10000)))
    mass_res.errorbar(x=x, xerr=dx, y=y - cp.breit_wigner(out.beta, x), yerr=dy, fmt='o')
    mass_res.plot(np.linspace(mass_res.get_xlim()[0], mass_res.get_xlim()[1], 10), np.zeros(10), color='C3')
    print(f'Pi-0 Mass Distribution')
    print(f'chi^2_red = {chisq_red:.2g}')
    print(f'P_value = {p_value:.2g}')
    print(f'Mass = {out.beta[2]} +- {out.sd_beta[2]:.2g} : '
          f'N_sigma = {cp.n_sigma((out.beta[2], out.sd_beta[2]), cp.pdg["pi-0"].mass)}')
    print(f'Gamma = {out.beta[3]} +- {out.sd_beta[3]:.2g} : '
          f'N_sigma = {cp.n_sigma((out.beta[3], out.sd_beta[3]), ((cp.pdg["pi-0"].tau[0] * cp.pdg["pi-0"].SEC) ** -1, cp.pdg["pi-0"].tau[1] / (cp.pdg["pi-0"].tau[0] ** 2 * cp.pdg["pi-0"].SEC)))}')

