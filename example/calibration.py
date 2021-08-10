"""
Calibration data collection and analysis
"""
import pickle
from pathlib import Path
import collipy as cp
import numpy as np
import pandas as pd
from scipy.constants import c, giga, centi
from collipy.helper import function_timer
import matplotlib.pyplot as plt


B_FACTOR = (2 * giga) / (c * centi)


@function_timer
def collect():
    """collecting `n` injections for each particle and momentum concurrently"""
    alpha = 1
    particles = ['electron', 'muon', 'photon']
    momenta = np.arange(1, 11, 1)
    modes = [cp.DecayMode(0, 1, 1), cp.DecayMode(0, 1, 1), cp.DecayMode(0, 0, 1)]
    n = 200
    threshold = 0.05
    ac = cp.Accelerator(alpha)
    data = {key: [] for key in particles}
    for particle, mode in zip(particles, modes):
        for momentum in momenta:
            data[particle].append(ac.collect(particle, momentum, n, mode, threshold))
    return data


def kappa_pt(lst: list[cp.InjectionCollection]) -> pd.DataFrame:
    """
    Extract kappas and tangential momenta from a list of 'cp.InjectionCollection' objects
    [electron with momentum 10 GeV data, electron with momentum 20 GeV data, ...]

    Parameters
    ----------
    lst
        [electron with momentum 10 GeV data, electron with momentum 20 GeV data, ...]
        each 'cp.InjectionCollection' object contain data of the same particle with different momentum

    Returns
    -------
    df
        DataFrame with the columns ['k', 'dk', 'pt', 'dpt']
    """
    rows = []
    for ic in lst:
        track = ic.get_df()[1]
        # adding statistical uncertainty to the measurements and calculating the mean
        dk = np.sqrt(track.dk ** 2 + np.var(track.k, ddof=1))
        k, sum = np.average(track.k, weights=dk ** -2, returned=True)
        dk = np.sqrt(sum ** -1)
        theta = np.arctan(track.tan_theta)
        dtheta = track.dtan_theta / (1 + track.tan_theta ** 2)
        p, dp = ic.momentum[0] * np.ones_like(track.tan_theta), ic.momentum[1] * np.ones_like(track.tan_theta)
        pt = p * np.cos(theta)
        dpt = np.abs(pt) * np.sqrt((dp / p) ** 2 + (np.tan(theta) * dtheta) ** 2)
        pt, sum = np.average(pt, weights=dpt ** -2, returned=True)
        dpt = np.sqrt(sum ** -1)
        rows.append([k, dk, pt, dpt])
    df = pd.DataFrame(rows, columns=['k', 'dk', 'pt', 'dpt'])
    return df


def ph_energy(lst: list[cp.InjectionCollection]) -> pd.DataFrame:
    """
    Extract pulse heights and energies from 'lst' a list of 'cp.InjectionCollection' objects
    note that this function is for particles with momentum that sustain: mass << momentum

    Parameters
    ----------
    lst
        [electron with momentum 10 GeV data, electron with momentum 20 GeV data, ...]
        each 'cp.InjectionCollection' object contain data of the same particle with different momentum

    Returns
    -------
    df
        DataFrame with the columns ['ph', 'dph', 'e', 'de']

    Notes
    -----
    This function suite for particles with momentum >> particle.mass

    """
    rows = []
    for ic in lst:
        ecal = ic.get_df()[2]
        n = len(ecal.ph)
        # no pulse height uncertainty given from software, estimated by the pulse height variance
        dph = np.std(ecal.ph, ddof=1) / np.sqrt(n)
        ph = np.mean(ecal.ph)
        e = ic.momentum[0]
        de = ic.momentum[1] / np.sqrt(n)
        rows.append([ph, dph, e, de])  # for momenta that meet mass<<momentum
    df = pd.DataFrame(rows, columns=['ph', 'dph', 'e', 'de'])
    return df


def calibrate_pt(data: {str: cp.InjectionCollection}, plot_it=False) -> dict:
    """Fitting data from 'kappa_pt' function

    Parameters
    ----------
    data:
        dictionary received from
    plot_it: bool
        whether to plot the fit or not

    Returns
    -------
        dict of the fitted parameters for each particle {particle name: (scipy.odr.Output, chi^2_reduced , p-value)}
    """
    if plot_it:
        _, fit = plt.subplots(subplot_kw=dict(title='Curvature - Transverse Momentum Calibration',
                                              ylabel=r'$\kappa[cm^{-1}]$',
                                              xlabel=r'$\frac{1}{p cos(\theta)}[GeV^{-1}]$'))
        _, res = plt.subplots(subplot_kw=dict(title='Curvature - Transverse Momentum Calibration Residuals Plot',
                                              ylabel=r'Residuals $[cm^{-1}]$',
                                              xlabel=r'$\frac{1}{p cos(\theta)}[GeV^{-1}]$'))
    b, db = [], []
    for particle in ['electron', 'muon']:
        df = kappa_pt(data[particle])
        y, dy = df.k, df.dk
        x, dx = 1 / df.pt, df.dpt / df.pt ** 2
        beta_guess = [1.3 / B_FACTOR, 0]
        out, chi, p_value = cp.fit(x, y, dy, lambda beta, x: np.poly1d(beta)(x), beta_guess)
        if plot_it:
            fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o', label=f'{particle.title()} Data')
            fit.plot(x, np.poly1d(out.beta)(x), label=f'{particle.title()} Fit')
            res.errorbar(x=x, xerr=dx, y=np.poly1d(out.beta)(x) - y, yerr=dy, fmt='o', label=f'{particle.title()}')
        b.append(B_FACTOR * out.beta[0])
        db.append(B_FACTOR * out.sd_beta[0])
        print(f'{particle.title()} Kappa-Momentum GOF')
        print(f'chi^2_red = {chi}')
        print(f'P_value = {p_value}')
        print(f'Beta: {out.beta}')
        print(f'Sd Beta: {out.sd_beta}')
        print(f'B_0 = {B_FACTOR * out.beta[0]:0.5g} +- {B_FACTOR * out.sd_beta[0]:0.5g} Tesla')
        print()
    avg, sum = np.average(b, weights=np.array(db) ** -2, returned=True)
    davg = np.sqrt(sum) ** -1
    print(f'B_0_avg = {avg:0.5g} +- {davg:0.2g} Tesla')
    print()
    if plot_it:
        fit.grid()
        fit.legend()
        res.plot(np.linspace(res.get_xlim()[0], res.get_xlim()[1], 10), np.zeros(10), color='C3')
        res.grid()
        res.legend()


def calibrate_energy(data: {str: cp.InjectionCollection}, plot_it=False) -> dict:
    """Fitting data from function 'ph_energy'

        Parameters
        ----------
        data:
            dictionary received from
        plot_it: bool
            whether to plot the fit or not

        Returns
        -------
            dict of the fitted parameters for each particle {particle name: (scipy.odr.Output, chi^2_reduced , p-value)}
        """
    if plot_it:
        _, fit = plt.subplots(subplot_kw=dict(title='Pulse Height - Energy Calibration',
                                              ylabel=r'$E[GeV]$', xlabel='Pulse Height'))
        _, res = plt.subplots(subplot_kw=dict(title='Pulse Height - Energy Calibration Residuals Plot',
                                              ylabel=r'Residuals $[GeV]$', xlabel='Pulse Height'))
    out_dic = {}
    for particle in ['electron', 'photon']:
        df = ph_energy(data[particle])
        y, dy = df.e, df.de
        x, dx = df.ph, df.dph
        beta_guess = [0.02, 0.2]
        out, chi, p_value = cp.fit(x, y, dy, lambda beta, x: np.poly1d(beta)(x), beta_guess, dx)
        out_dic[particle] = out, chi, p_value
        if plot_it:
            fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o', label=f'{particle.title()} Data')
            fit.plot(x, np.poly1d(out.beta)(x), label=f'{particle.title()} Fit')
            res.errorbar(x=x, xerr=dx, y=np.poly1d(out.beta)(x) - y, yerr=dy, fmt='o', label=f'{particle.title()}')
        print(f'{particle.title()} Pulse Height-Energy GOF')
        print(f'chi^2_red = {chi}')
        print(f'P_value = {p_value}')
        print(f'Beta: {out.beta}')
        print(f'Sd Beta: {out.sd_beta}')
        print()
    if plot_it:
        fit.grid()
        fit.legend()
        res.plot(np.linspace(res.get_xlim()[0], res.get_xlim()[1], 10), np.zeros(10), color='C3')
        res.grid()
        res.legend()
    return out_dic


if __name__ == '__main__':
    # loading/creating data
    path = Path('data/calibration_new.pickle')
    folder = 'data'
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    calibrate_pt(data)
    calibrate_energy(data)
