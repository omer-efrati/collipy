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
    n = 2_400
    mode = cp.DecayMode(1, 2, None)
    threshold = np.inf
    ac = cp.Accelerator(alpha)
    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    return data


def cond(inj):
    threshold = 0.05
    # opposite charged particles
    if np.sign(inj.track.k.iloc[0]) != np.sign(inj.track.k.iloc[1]):
        if np.max(np.abs(inj.vertex.dx / inj.vertex.x)) <= threshold:
            if np.max(
                    np.abs([inj.track.dk / inj.track.k, inj.track.dtan_theta / inj.track.tan_theta])) <= threshold:
                if np.max(np.abs(inj.vertex.dphi / inj.vertex.phi)) <= threshold:
                    return True
    return False


def pt(k, sd_k, alpha):
    a, b = alpha * 1.94725051e-03, 3.38959156e-07
    sd_a, sd_b = alpha * 1.06133148e-06, 3.13139306e-07
    res = a / (k - b)
    sd_res = np.abs(res) * np.sqrt((sd_a/a)**2 + (sd_k/(k-b))**2 + (sd_b/(k-b))**2)
    return res, sd_res


def get_momentum(data) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # the curvature is positive for negative charged particle and vice versa
    track = data.get_df()[1]
    plus = track.query('k < 0')
    minus = track.query('k > 0')
    pt1, dpt1 = pt(plus.k, plus.dk, data.alpha)
    pt2, dpt2 = pt(minus.k, minus.dk, data.alpha)
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


def get_time(data):
    df = data.get_df()[0]
    momentum = data.momentum
    time, dtime = [], []
    for _, row in df.iterrows():
        r, dr = get_r(row)
        r, dr = r/100, dr/100
        r, dr = cp.pdg['k-short'].METER * r, cp.pdg['k-short'].METER * dr
        m, dm = cp.pdg['k-short'].mass
        p, dp = momentum
        t = (m * r) / p
        dt = t * np.sqrt((dm/m)**2 + (dr/r)**2 + (dp/p)**2)
        t, dt = t/cp.pdg['k-short'].SEC, dt/cp.pdg['k-short'].SEC
        time.append(t)
        dtime.append(dt)
    return np.array(time), np.array(dtime)


def mass():
    with open('data/kshort15%.pickle', 'rb') as file:
        data = pickle.load(file)
    data.cnt['total'] -= data.cnt[(0, 0, 0)]
    del data.cnt[(0, 0, 0)]

    ni = np.array(
        [val for key, val in data.cnt.items() if isinstance(key, tuple) and not (key[0] == 0 and key[1] == 0)])
    N = data.cnt['total']
    br = np.sum(ni) / N
    sd_br = np.sqrt((np.sqrt(np.sum(ni)) / N) ** 2 + (br / np.sqrt(N)) ** 2)
    print(f'Branching Ratio for K-short = {br} +- {sd_br}')

    vertex, _, _ = data.get_df()
    m1, dm1 = cp.pdg['pi-plus'].mass
    m2, dm2 = cp.pdg['pi-minus'].mass
    p1, dp1, p2, dp2 = get_momentum(data)
    phi, dphi = get_phi(vertex)

    one = m1 ** 2 + m2 ** 2
    done = np.sqrt((2 * m1 * dm1) ** 2 + (2 * m2 * dm2) ** 2)
    two = 2 * np.sqrt((m1 ** 2 + p1 ** 2) * (m2 ** 2 + p2 ** 2))
    dtwo = np.abs(two) * np.sqrt(((2 * m1 * dm1) / (m1 ** 2 + p1 ** 2)) ** 2
                                 + ((2 * p1 * dp1) / (m1 ** 2 + p1 ** 2)) ** 2
                                 + ((2 * m2 * dm2) / (m2 ** 2 + p2 ** 2)) ** 2
                                 + ((2 * p2 * dp2) / (m2 ** 2 + p2 ** 2)) ** 2)
    three = -2 * np.abs(p1 * p2) * np.cos(phi)
    dthree = np.abs(three) * np.sqrt((dp1 / p1) ** 2 + (dp2 / p2) ** 2 + (np.tan(phi) * dphi) ** 2)
    msq = one + two + three
    dmsq = np.sqrt(done ** 2 + dtwo ** 2 + dthree ** 2)
    m = np.sqrt(msq)
    dm = dmsq / (2 * m)
    y, x = np.histogram(m, bins=10, range=(min(m), 2 * cp.pdg['k-short'].mass[0] - min(m)))
    dy = np.sqrt(y)
    dx = np.full_like(y, (x[1] - x[0]) / 2, dtype=float)
    x = np.array([(x[i - 1] + x[i]) / 2 for i in range(1, len(x))])
    beta_initial = [160, cp.pdg['k-short'].mass[0], cp.pdg['k-short'].tau[0]*1e6]
    out, chisq_red, p_value = cp.fit(x, y, dy, cp.breit_wigner, beta_initial, dx)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3, 1]))
    fig.suptitle(r'$K^0_s$ Mass Distribution')
    fig.supxlabel(r'$m[GeV]$')
    ax1.set_ylabel(r'Events')
    ax1.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
    xs = np.linspace(0.3, 2 * out.beta[1] - 0.3, 10_000)
    ax1.plot(xs, cp.breit_wigner(out.beta, xs))
    ax1.grid()
    ax2.set_ylabel('$residuals$')
    ax2.errorbar(x=x, xerr=dx, y=y - cp.breit_wigner(out.beta, x), yerr=dy, fmt='o')
    ax2.axhline(0, color='C3')
    ax2.grid()
    print(f'K0_s Mass Distribution')
    print(f'chi^2_red = {chisq_red}')
    print(f'P_value = {p_value}')
    print(f'Mass = {out.beta[1]} +- {out.sd_beta[1]} : '
          f'N_sigma = {cp.n_sigma((out.beta[1], out.sd_beta[1]), cp.pdg["k-short"].mass)}')
    print(f'Gamma = {out.beta[2]} +- {out.sd_beta[2]} : '
          f'N_sigma = {cp.n_sigma((out.beta[2], out.sd_beta[2]), ((cp.pdg["k-short"].tau[0] * cp.pdg["k-short"].SEC) ** -1, cp.pdg["k-short"].tau[1] / (cp.pdg["k-short"].tau[0] ** 2 * cp.pdg["k-short"].SEC)))}')


def lifetime():
    with open('data/kshort_lifetime.pickle', 'rb') as file:
        data = pickle.load(file)
    t, dt = get_time(data)
    # t, dt = t[:200], dt[:200]
    y, x = np.histogram(t, bins=6, range=(5.3e-11, 2e-10))
    dy = np.sqrt(y)
    dx = ((x[1] - x[0]) / 2) * np.ones_like(y)
    x = np.array([(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)])
    beta_initial = [np.max(y) / np.exp(-x[np.argmax(y)] / cp.pdg['k-short'].tau[0]), cp.pdg['k-short'].tau[0], 0]
    out, chisq_red, p_value = cp.fit(x, y, dy, cp.expon, beta_initial, dx)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3, 1]))
    fig.suptitle(r'$K^0_s$ Lifetime Distribution')
    fig.supxlabel(r'$\tau[s]$')
    ax1.set_ylabel(r'Events')
    ax1.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
    xs = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 10_000)
    ax1.plot(xs, cp.expon(out.beta, xs))
    ax1.grid()
    ax2.set_ylabel('$residuals$')
    ax2.errorbar(x=x, xerr=dx, y=y - cp.expon(out.beta, x), yerr=dy, fmt='o')
    ax2.axhline(0, color='C3')
    ax2.grid()
    print(f'K0_s Lifetime Distribution')
    print(f'chi^2_red = {chisq_red}')
    print(f'P_value = {p_value}')
    print(f'Tau = {out.beta[1]} +- {out.sd_beta[1]} : '
          f'N_sigma = {cp.n_sigma((out.beta[1], out.sd_beta[1]), cp.pdg["k-short"].tau)}')

    # print(f'K0_s Lifetime Distribution')
    # print(f'chi^2_red = {chisq_red}')
    # print(f'P_value = {p_value}')
    # print(f'Tau = {out.beta[1]} +- {out.sd_beta[1]} : '
    #       f'N_sigma = {cp.n_sigma((out.beta[1], out.sd_beta[1]), cp.pdg["k-short"].tau)}')


if __name__ == '__main__':
    lifetime()