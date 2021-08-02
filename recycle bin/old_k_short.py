import logging.config
import os
import numpy as np
import pandas as pd
from collider import Collider, Data, DecayMode
from simulator.particles import pi_minus, pi_plus
from examples.calibration import Cal
import matplotlib.pyplot as plt

logging.config.fileConfig('../log/logging.conf')
logger = logging.getLogger('kShortLog')

""" The idea is to inject k-short with the right momentum so it will have high chance to decay far from the ECAL
    and inside the magnetic calorimeter. therefore the decay mode we expect is DecayMode(1,2,0)."""

particle = 'k-short'
filename = f'{particle}.pickle'
path = os.path.join('../data', filename)
if filename not in os.listdir('../data'):
    user = input('username: \n')
    password = input('password: \n')
    n = 1000
    threshold = 0.25
    alpha = 10
    # empirically best momentum to collect the required data
    momentum = 3.7
    mode = DecayMode(1, 2, 0)
    c = Collider(user, password, alpha)
    logger.info(f'Injecting {particle} {momentum} GeV')
    data = c.collect(particle, momentum, n, threshold, mode)
    logger.info(f'Finished injecting {particle} {momentum} GeV')
    data.dump(path)
else:
    data = Data.load(path)

br = np.sum([val for key, val in data.cnt.items() if key[2] == 0]) / data.m


def get_momentum(track: pd.DataFrame, kappa_pt) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # the curvature is positive for negative charged particle and vice versa
    plus = track.query('k < 0')
    minus = track.query('k > 0')
    pt1, dpt1 = kappa_pt(plus.k, plus.dk)
    pt2, dpt2 = kappa_pt(minus.k, minus.dk)
    theta1, dtheta1 = np.arctan(plus.tan_theta), plus.dtan_theta / (1 + plus.tan_theta ** 2)
    theta2, dtheta2 = np.arctan(minus.tan_theta), minus.dtan_theta / (1 + minus.tan_theta ** 2)
    p1 = pt1/np.cos(theta1)
    dp1 = np.abs(p1) * np.sqrt((dpt1/pt1)**2 + np.abs(np.tan(theta1))*dtheta1)
    p2 = pt2/np.cos(theta2)
    dp2 = np.abs(p2) * np.sqrt((dpt2/pt2)**2 + np.abs(np.tan(theta2))*dtheta2)
    return p1.to_numpy(), dp1.to_numpy(), p2.to_numpy(), dp2.to_numpy()


def get_phi(vertex: pd.DataFrame) -> (np.ndarray, np.ndarray):
    return vertex.phi.to_numpy(), vertex.dphi.to_numpy()

c = Cal(data.alpha)
index = []
for i, t in data.track.groupby(level=0):
    if t.k.iloc[0] * t.k.iloc[1] < 0:
        index.append(i)
track = data.track.loc[index]
vertex = data.vertex.loc[index]
m1, dm1 = pi_plus.mass
m2, dm2 = pi_minus.mass
p1, dp1, p2, dp2 = get_momentum(track, c.kappa_pt)
phi, dphi = get_phi(vertex)

one = m1**2 + m2**2
done = np.sqrt((2 * m1 * dm1)**2 + (2 * m2 * dm2)**2)
two = 2 * np.sqrt((m1**2 + p1**2) * (m2**2 + p2**2))
dtwo = two * np.sqrt(((2*m1*dm1) / (m1**2 + p1**2))**2 + ((2*p1*dp1) / (m1**2 + p1**2))**2 +
                     ((2*m2*dm2) / (m2**2 + p2**2))**2 + ((2*p2*dp2) / (m2**2 + p2**2))**2)
three = -2 * np.abs(p1 * p2) * np.cos(phi)
dthree = np.abs(three) * np.sqrt((dp1/p1)**2 + (dp2/p2)**2 + (np.tan(phi)*dphi)**2)
msq = one + two + three
dmsq = np.sqrt(done**2 + dtwo**2 + dthree**2)
m = np.sqrt(msq)
dm = dmsq/(2*m)
w = dm**-2
bins = 13
y, x = np.histogram(a=m, bins=bins, range=(0.3, 0.65), weights=w)
dy = np.sqrt(np.histogram(a=m, bins=bins, range=(0.3, 0.65), weights=w**2)[0])
dx = ((x[1]-x[0]) / 2) * np.ones_like(y)
x = np.array([(x[i] + x[i+1])/2 for i in range(len(x)-1)])
plt.errorbar(x=x, xerr=dx, y=y, yerr=dy, fmt='o')
# z = np.abs(zscore(msq))
# msq = msq[z < 3]  # throwing outliers
