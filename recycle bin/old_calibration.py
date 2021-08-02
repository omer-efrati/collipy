import logging
import logging.config
import os
import pickle

import numpy as np
import pandas as pd
from scipy.constants import c, giga, centi
from collider import Collider, Data, DecayMode
from util import fit_ols
import matplotlib.pyplot as plt

logging.config.fileConfig('../log/logging.conf')
logger = logging.getLogger('callibrationLog')
trans = (2 * giga) / (c * centi)


def collect_data(particles: {str: DecayMode}, momenta: list, filename='cal.pickle', n=200, threshold=0.05) -> dict:
    """Collecting data for calibration
        notice that if the file exists, the function will load the data from it
    Parameters
    ----------
    particles
        {key: value} where the key is a particle and the value is the decay mode required
        example: {'electron': DecayMode(0,1,1)} means injecting electron and receiving a reading
        no vertex, single track and a single reading at the ECAL
    momenta
        list of momentum for each particle
    filename: str
        name of file to load the data from or save the data to after collecting
    n: int
        lower bound for the number of measurements required
    threshold: float
        an upper bound for the relative uncertainty allowed for all the parameters in a single injection
        vertex.dx/vertex.x, ..., track.dk/track.k, ..., ecal.dph/ecal.ph, ... <= 'threshold'
    Returns
    -------
        res
            {particle name: 'Data' object for each momentum from 'momenta'}
    """
    path = os.path.join('../example/data', filename)
    if filename not in os.listdir('../example/data'):
        user = input('username: \n')
        password = input('password: \n')
        data = {key: [] for key in particles.keys()}
        for particle, mode in particles.items():
            # new collider for each particle to avoid getting kicked out from the server (may happen after a while)
            c = Collider(user, password)
            for momentum in momenta:
                data[particle].append(c.collect(particle, momentum, n, threshold, mode))
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not have to specify it.
            data = pickle.load(f)
    return data


def kappa_pt(lst: [Data]) -> pd.DataFrame:
    """Calculate the data for kappa-momentum calibration from a list of 'Data' objects

    Parameters
    ----------
    lst
        each 'Data' object contain data of the same particle with different momentum

    Returns
    -------
    df
        DataFrame with the columns ['k', 'dk', 'pt', 'dpt']
    """
    rows = []
    for data in lst:
        track = data.track
        # adding statistical uncertainty to the measurements and calculating the mean
        dk = np.sqrt(track.dk ** 2 + np.var(track.k, ddof=1))
        k, sum = np.average(track.k, weights=dk ** -2, returned=True)
        dk = np.sqrt(sum ** -1)
        theta = np.arctan(track.tan_theta)
        dtheta = track.dtan_theta / (1 + track.tan_theta ** 2)
        p, dp = data.momentum[0] * np.ones_like(track.tan_theta), data.momentum[1] * np.ones_like(track.tan_theta)
        pt = p * np.cos(theta)
        dpt = pt * np.sqrt((dp / p) ** 2 + (np.tan(theta) * dtheta) ** 2)
        dpt = np.sqrt(dpt ** 2 + np.var(pt, ddof=1))
        pt, sum = np.average(pt, weights=dpt ** -2, returned=True)
        dpt = np.sqrt(sum ** -1)
        rows.append([k, dk, pt, dpt])
    df = pd.DataFrame(rows, columns=['k', 'dk', 'pt', 'dpt'])
    return df


def ph_energy(lst: [Data]) -> pd.DataFrame:
    """Calculate the data for pulse height-energy calibration from a list of 'Data' objects
        note that this function is for particles with momentum that sustain: mass << momentum

        Parameters
        ----------
        lst
            each 'Data' object contain data of the same particle with different momentum

        Returns
        -------
        df
            DataFrame with the columns ['ph', 'dph', 'e', 'de']
        """
    # for particles with mass << momentum
    rows = []
    for data in lst:
        ecal = data.ecal
        dph = np.sqrt(ecal.dph ** 2 + np.var(ecal.ph, ddof=1))
        ph, sum = np.average(ecal.ph, weights=dph ** -2, returned=True)
        dph = np.sqrt(sum ** -1)
        de = data.momentum[1] * np.ones_like(ecal.ph)
        e, sum = np.average(data.momentum[0] * np.ones_like(ecal.ph), weights=de ** -2, returned=True)
        de = np.sqrt(sum ** -1)
        rows.append([ph, dph, data.momentum[0], de])  # for momenta that meet mass<<momentum
    df = pd.DataFrame(rows, columns=['ph', 'dph', 'e', 'de'])
    return df


def calibrate_pt(data: {str: Data}, plot_it=False) -> dict:
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
        _, fit = plt.subplots(subplot_kw=dict(title=f'Kappa - Momentum Calibration',
                                              ylabel=r'$\kappa[cm^{-1}]$',
                                              xlabel=r'$\frac{1}{p cos(\theta)}[GeV^{-1}]$'))
        _, res = plt.subplots(subplot_kw=dict(title=f'Kappa - Momentum Calibration Residuals Plot',
                                              ylabel=r'$f(p_i, \theta_i) - \kappa_i$',
                                              xlabel=r'$\frac{1}{p}[GeV^{-1}]$'))
    out_dic = {}
    b = []
    db = []
    for particle in ['electron', 'muon']:
        df = kappa_pt(data[particle])
        y, dy = df.k, df.dk
        x, dx = 1 / df.pt, df.dpt / df.pt ** 2
        beta_guess = [1.3 / trans, 0]
        out, chi, p_value = fit_ols(x, dx, y, dy, lambda beta, x: np.poly1d(beta)(x), beta_guess)
        out_dic[particle] = out, chi, p_value
        if plot_it:
            fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o', label=f'{particle.title()} Data')
            fit.plot(out.xplus, out.y, label=f'{particle.title()} Fit')
            res.errorbar(x=out.xplus, xerr=dx, y=out.eps, yerr=dy, fmt='o', label=f'{particle.title()}')
        b.append(trans * out.beta[0])
        db.append(trans * out.sd_beta[0])
        print(f'{particle.title()} Kappa-Momentum GOF')
        print(f'chi^2_red = {chi}')
        print(f'P_value = {p_value}')
        print(f'Beta: {out.beta}')
        print(f'Sd Beta: {out.sd_beta}')
        print(f'B_0 = {trans * out.beta[0]:0.5g} +- {trans * out.sd_beta[0]:0.5g} Tesla')
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
    return out_dic


def calibrate_energy(data: {str: Data}, plot_it=False) -> dict:
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
        _, fit = plt.subplots(subplot_kw=dict(title=f'Pulse Height - Energy Calibration',
                                              ylabel=r'$E[GeV]$', xlabel=r'$ph$'))
        _, res = plt.subplots(subplot_kw=dict(title=f'Pulse Height - Energy Calibration Residuals Plot',
                                              ylabel=r'$f(ph_i) - E_i[GeV]$', xlabel=r'$ph$'))
    out_dic = {}
    for particle in ['electron', 'photon']:
        df = ph_energy(data[particle])
        y, dy = df.e, df.de
        x, dx = df.ph, df.dph
        beta_guess = [1, 1]
        out, chi, p_value = fit_ols(x, dx, y, dy, lambda beta, x: np.poly1d(beta)(x), beta_guess)
        out_dic[particle] = out, chi, p_value
        if plot_it:
            fit.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o', label=f'{particle.title()} Data')
            fit.plot(out.xplus, out.y, label=f'{particle.title()} Fit')
            res.errorbar(x=out.xplus, xerr=dx, y=out.eps, yerr=dy, fmt='o', label=f'{particle.title()}')
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


def best_fit(outs):
    """Choosing best fit"""
    best = 0
    for key, (_, _, p_value) in outs.items():
        if abs(p_value-0.5) < abs(best-0.5):
            best_key = key
            best = p_value
    return best_key, outs[best_key]


class Cal:
    """Containing information and function about the collider after calibration
        DO NOT USE THIS CLASS BEFORE CALIBRATING THE COLLIDER"""
    def __init__(self, alpha: float):
        filename = 'cal_func.pickle'
        if filename not in os.listdir('../example/data'):
            raise Exception("Calibration is needed before creating a 'Cal' object")
        with open(os.path.join('../example/data', filename), 'rb') as f:
            params = pickle.load(f)
        self.alpha = alpha
        self._momentum_beta = params['momentum']
        self._energy_beta = params['energy']

    @property
    def momentum_beta(self):
        beta = np.array(self._momentum_beta[0])
        dbeta = np.array(self._momentum_beta[1])
        beta[0] *= self.alpha
        dbeta[0] *= self.alpha
        return beta, dbeta

    @property
    def energy_beta(self):
        beta = np.array(self._energy_beta[0])
        dbeta = np.array(self._energy_beta[1])
        return beta, dbeta

    def kappa_pt(self, k: list, dk: list) -> (list, list):
        """calibration function for kappa-momentum
            ***note you can use this function only after calibration***
        Parameters
        ----------
        k
            list of kappas
        dk
            list of 'k' corresponding uncertainties

        Returns
        -------
            pt, dpt: list, list
        """
        beta = self.momentum_beta[0]
        dbeta = self.momentum_beta[1]
        pt = beta[0] / (k - beta[1])
        dpt = np.abs(pt) * np.sqrt((dbeta[0]/beta[0])**2 + (dk / (k - beta[1]))**2 + (dbeta[1] / (k - beta[1]))**2)
        return pt, dpt

    def ph_e(self, ph: list, dph: list) -> (list, list):
        """calibration function for pulse height-energy
            ***note you can use this function only after calibration***
        Parameters
        ----------
        ph
            list of pulse heights
        dph
            list of 'ph' corresponding uncertainties

        Returns
        -------
            e, de: list, list
        """
        if not self._energy_beta:
            raise Exception('Calibration is needed first')
        beta = self._energy_beta[0]
        dbeta = self._energy_beta[1]
        e = np.poly1d(beta)(ph)
        de = np.sqrt((dbeta[0]*ph)**2 + (beta[0]*dph)**2 + (dbeta[1])**2)
        return e, de


if __name__ == '__main__':
    filename = 'cal_func.pickle'
    particles = {'electron': DecayMode(0, 1, 1), 'muon': DecayMode(0, 1, 1), 'photon': DecayMode(0, 0, 1)}
    data = collect_data(particles, [i for i in range(10, 110, 10)])

    out_pt = calibrate_pt(data)
    out_e = calibrate_energy(data)
    particle_pt, best_pt = best_fit(out_pt)
    print(f'Kappa-Momentum Calibration: {particle_pt} with Chi^2_red={best_pt[1]:0.2g} P-value={best_pt[2]:0.2g}')
    particle_e, best_e = best_fit(out_e)
    print(f'Pulse Height-Energy Calibration: {particle_e} with Chi^2_red={best_e[1]:0.2g} P-value={best_e[2]:0.2g}')

    cal = {'momentum': (best_pt[0].beta, best_pt[0].sd_beta), 'energy': (best_e[0].beta, best_e[0].sd_beta)}
    with open(os.path.join('../example/data', filename), 'wb') as f:
        pickle.dump(cal, f, pickle.HIGHEST_PROTOCOL)

