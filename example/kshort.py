"""
K-short data collection and analysis
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
    alpha = 8
    particle = 'k-short'
    momentum = 3.7
    n = 20_000
    mode = cp.DecayMode(1, 2, None)
    threshold = 100
    ac = cp.Accelerator(alpha)
    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    return data


def cond_decay(inj):
    """
    Injection decay verification for K-short

    Parameters
    ----------
    inj : cp.Injection

    Returns
    -------
        bool
            true for valid injection
    """
    if inj.track.k.iloc[0] * inj.track.k.iloc[1] < 0:
        return True
    return False


def cond_mass(threshold):
    """
    Injection precision condition for mass measurement

    Parameters
    ----------
    inj : cp.Injection

    Returns
    -------
        function
            a function that returns true for valid injection
    """
    def cond(inj):
        if cond_decay(inj):
            if np.max(np.abs([inj.track.dk / inj.track.k, inj.track.dtan_theta / inj.track.tan_theta])) <= threshold:
                return True
        return False

    return cond


def cond_tau(threshold):
    """
    Injection precision condition for lifetime measurement

    Parameters
    ----------
    inj : cp.Injection

    Returns
    -------
        function
            a function that returns true for valid injection
    """
    def cond(inj):
        if cond_decay(inj):
            if np.max(np.abs(inj.vertex.dx / inj.vertex.x)) <= threshold:
                return True
        return False

    return cond


def pt(k, sd_k):
    alpha = 10
    a, b = [1.94104212e-03, 2.16085896e-07]
    a *= alpha
    sd_a, sd_b = [6.58324160e-06, 2.53692179e-07]
    sd_a *= 10
    res = a / (k - b)
    sd_res = np.abs(res) * np.sqrt((sd_a/a)**2 + (sd_k/(k-b))**2 + (sd_b/(k-b))**2)
    return res, sd_res


def get_momentum(track: pd.DataFrame, pt) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # the curvature is positive for negative charged particle and vice versa
    plus = track.query('k < 0')
    minus = track.query('k > 0')
    pt1, dpt1 = pt(plus.k, plus.dk)
    pt2, dpt2 = pt(minus.k, minus.dk)
    theta1, dtheta1 = np.arctan(plus.tan_theta), plus.dtan_theta / (1 + plus.tan_theta ** 2)
    theta2, dtheta2 = np.arctan(minus.tan_theta), minus.dtan_theta / (1 + minus.tan_theta ** 2)
    p1 = pt1/np.cos(theta1)
    dp1 = np.abs(p1) * np.sqrt((dpt1/pt1)**2 + np.abs(np.tan(theta1))*dtheta1)
    p2 = pt2/np.cos(theta2)
    dp2 = np.abs(p2) * np.sqrt((dpt2/pt2)**2 + np.abs(np.tan(theta2))*dtheta2)
    return p1.to_numpy(), dp1.to_numpy(), p2.to_numpy(), dp2.to_numpy()


def get_phi(vertex: pd.DataFrame) -> (np.ndarray, np.ndarray):
    return vertex.phi.to_numpy(), vertex.dphi.to_numpy()


def get_r(vertex) -> (float, float):
    """Get distance from point of injection to vertex
    *** valid function for injections with a single vertex only"""
    u = np.array([vertex.x, vertex.y, vertex.z])
    du = np.array([vertex.dx, vertex.dy, vertex.dz])
    r = np.linalg.norm(u)
    dr = np.linalg.norm(u * du) / r
    return r, dr


def get_time(mass, momentum, df: pd.DataFrame):
    time, dtime = [], []
    for _, row in df.iterrows():
        r, dr = get_r(row)
        r, dr = r/100, dr/100
        r, dr = cp.pdg['k-short'].METER * r, cp.pdg['k-short'].METER * dr
        m, dm = mass
        p, dp = momentum
        t = (m * r) / p
        dt = t * np.sqrt((dm/m)**2 + (dr/r)**2 + (dp/p)**2)
        t, dt = t/cp.pdg['k-short'].SEC, dt/cp.pdg['k-short'].SEC
        time.append(t)
        dtime.append(dt)
    return np.array(time), np.array(dtime)


def main():
    path = Path('data/kshort.pickle')
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()

    # deleting unknown decays
    data.cnt['total'] -= data.cnt[(0, 0, 0)]
    del data.cnt[(0, 0, 0)]

    ni = np.array(
        [val for key, val in data.cnt.items() if isinstance(key, tuple) and not (key[0] == 0 and key[1] == 0)])
    N = data.cnt['total']
    br = np.sum(ni) / N
    sd_br = np.sqrt((np.sqrt(np.sum(ni)) / N) ** 2 + (br / np.sqrt(N)) ** 2)
    print(f'Branching Ratio for K-short = {br:.5g} +- {sd_br:.2g}')

    vertex, track, _ = data.get_df()
    m1, dm1 = cp.pdg['pi-plus'].mass
    m2, dm2 = cp.pdg['pi-minus'].mass
    p1, dp1, p2, dp2 = get_momentum(track, pt)
    phi, dphi = get_phi(vertex)

    one = m1 ** 2 + m2 ** 2
    done = np.sqrt((2 * m1 * dm1) ** 2 + (2 * m2 * dm2) ** 2)
    two = 2 * np.sqrt((m1 ** 2 + p1 ** 2) * (m2 ** 2 + p2 ** 2))
    dtwo = two * np.sqrt(((2 * m1 * dm1) / (m1 ** 2 + p1 ** 2)) ** 2 + ((2 * p1 * dp1) / (m1 ** 2 + p1 ** 2)) ** 2 +
                         ((2 * m2 * dm2) / (m2 ** 2 + p2 ** 2)) ** 2 + ((2 * p2 * dp2) / (m2 ** 2 + p2 ** 2)) ** 2)
    three = -2 * np.abs(p1 * p2) * np.cos(phi)
    dthree = np.abs(three) * np.sqrt((dp1 / p1) ** 2 + (dp2 / p2) ** 2 + (np.tan(phi) * dphi) ** 2)
    msq = one + two + three
    dmsq = np.sqrt(done ** 2 + dtwo ** 2 + dthree ** 2)
    m = np.sqrt(msq)
    dm = dmsq / (2 * m)
    w = dm ** -2
    bins = 11

    y, x = np.histogram(a=m, bins=bins, range=(0.344, 0.65), weights=w)
    dy = np.sqrt(np.histogram(a=m, bins=bins, range=(0.344, 0.65), weights=w ** 2)[0])
    dx = ((x[1] - x[0]) / 2) * np.ones_like(y)
    x = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    x, dx, y = x[30 < dy], dx[30 < dy], y[30 < dy]
    dy = dy[30 < dy]
    beta_initial = [0, 1, cp.pdg['k-short'].mass[0], 0.1]
    out, chisq_red, p_value = cp.fit(x, y, dy, cp.breit_wigner, beta_initial)

    _, mass_fit = plt.subplots(subplot_kw=dict(title=f'$K^0_s$ Mass Distribution',
                                               ylabel=r'Weighted Events', xlabel=r'$Mass[GeV]$'))
    _, mass_res = plt.subplots(subplot_kw=dict(title=f'$K^0_s$ Mass Distribution Residuals Plot',
                                               ylabel=r'Weighted Events Residuals', xlabel=r'$Mass[GeV]$'))
    mass_fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
    mass_fit.plot(np.linspace(0.344, 0.65, 10_000), cp.breit_wigner(out.beta, np.linspace(0.344, 0.65, 10000)))
    mass_res.errorbar(x=x, xerr=dx, y=y - cp.breit_wigner(out.beta, x), yerr=dy, fmt='o')
    mass_res.plot(np.linspace(mass_res.get_xlim()[0], mass_res.get_xlim()[1], 10), np.zeros(10), color='C3')
    print(f'K0_s Mass Distribution')
    print(f'chi^2_red = {chisq_red:.2g}')
    print(f'P_value = {p_value:.2g}')
    print(f'Mass = {out.beta[2]} +- {out.sd_beta[2]:.2g} : '
          f'N_sigma = {cp.n_sigma((out.beta[2], out.sd_beta[2]), cp.pdg["k-short"].mass)}')
    print(f'Gamma = {out.beta[3]} +- {out.sd_beta[3]:.2g} : '
          f'N_sigma = {cp.n_sigma((out.beta[3], out.sd_beta[3]), ((cp.pdg["k-short"].tau[0] * cp.pdg["k-short"].SEC) ** -1, cp.pdg["k-short"].tau[1] / (cp.pdg["k-short"].tau[0] ** 2 * cp.pdg["k-short"].SEC)))}')

    # TAU
    t, dt = get_time(cp.pdg['k-short'].mass, data.momentum, data.get_df()[0])
    z = np.abs(stats.zscore(t))
    # throwing outliers
    t, dt = t[z < 3], dt[z < 3]
    w = dt ** -2
    bins = 7
    y, x = np.histogram(a=t, bins=bins)
    dy = np.sqrt(y)
    dx = ((x[1] - x[0]) / 2) * np.ones_like(y)
    x = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    x, dx, y = x[0 < dy], dx[0 < dy], y[0 < dy]
    dy = dy[0 < dy]
    beta_initial = [np.max(y) / np.exp(-x[np.argmax(y)] / cp.pdg['k-short'].tau[0]), cp.pdg['k-short'].tau[0], 0]
    out, chisq_red, p_value = cp.fit(x, y, dy, cp.expon, beta_initial, dx)
    _, tau_fit = plt.subplots(subplot_kw=dict(title=f'$K^0_s$ Lifetime Distribution',
                                              ylabel=r'Events', xlabel=r'$\tau[s]$'))
    _, tau_res = plt.subplots(subplot_kw=dict(title=f'$K^0_s$ Lifetime Distribution Residuals Plot',
                                              ylabel=r'$f(Event_i) - Event_i$', xlabel=r'$\tau[s]$'))
    tau_fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
    xs = np.linspace(tau_fit.get_xlim()[0], tau_fit.get_xlim()[1], 10_000)
    tau_fit.plot(xs, cp.expon(out.beta, xs))
    tau_res.errorbar(x=x, xerr=dx, y=y - cp.expon(out.beta, x), yerr=dy, fmt='o')
    tau_res.plot(np.linspace(tau_res.get_xlim()[0], tau_res.get_xlim()[1], 10), np.zeros(10), color='C3')
    print(f'K0_s Lifetime Distribution')
    print(f'chi^2_red = {chisq_red:.2g}')
    print(f'P_value = {p_value:.2g}')
    print(f'Tau = {out.beta[1]} +- {out.sd_beta[1]:.2g} : '
          f'N_sigma = {cp.n_sigma((out.beta[1], out.sd_beta[1]), cp.pdg["k-short"].tau):.2g}')


if __name__ == '__main__':
    main()