import logging
import logging.config
import os
import pickle
from collider import Collider, Data, DecayMode


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dataLog')
user = input('username: \n')
password = input('password: \n')


def calibration():
    filename = 'cal.pickle'
    n = 200
    threshold = 0.05
    momenta = [i for i in range(10, 110, 10)]
    path = os.path.join('data', filename)
    if filename not in os.listdir('data'):
        data = {key: [] for key in ['electron', 'muon', 'photon']}
        particles = {'electron': DecayMode(0, 1, 1), 'muon': DecayMode(0, 1, 1), 'photon': DecayMode(0, 0, 1)}
        for particle, mode in particles.items():
            # After couple of hours of open connection the server kicks us out
            # so i refresh the connection once in a while
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


def particle_by_mode(alpha: float, particle: str, momentum: float, mode: DecayMode):
    filename = f'{particle}.pickle'
    n = 1000
    threshold = 0.25
    path = os.path.join('data', filename)
    if filename not in os.listdir('data'):
        c = Collider(user, password, alpha)
        logger.info(f'Injecting {particle} {momentum} GeV')
        data = c.collect(particle, momentum, n, threshold, mode)
        logger.info(f'Finished injecting {particle} {momentum} GeV')
        data.dump(path)
    else:
        data = Data.load(path)
    return data


if __name__ == '__main__':
    cal = calibration()
    ks = particle_by_mode(10, 'k-short', 3.7, DecayMode(1, 2, 0))
