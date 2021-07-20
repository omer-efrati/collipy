from server import GSH
import pandas as pd
import numpy as np


class Collider:

    def __init__(self, user: str, password: str, alpha=1):
        low = 0.1
        high = 10
        if not low <= alpha <= high:
            raise ValueError(f"alpha must meet the condition: {low} <= alpha <= {high}")
        self.alpha = alpha
        self.geant = GSH(user, password, alpha)

    def inject(self, particle: str, momentum: float, n: int) -> list:
        """Injecting n particles and returns a list of Injection objects for successful injections
        therefore the length of the list is $n at most (may be less)"""
        lst = self.geant.inject(particle, momentum, n)
        res = []
        for tmp in lst:
            res.append(Injection(tmp[0], tmp[1], tmp[2]))
        return InjectionsCurator(particle, momentum, res)


class Injection:

    def __init__(self, vertex: pd.DataFrame, track: pd.DataFrame, ecal: pd.DataFrame):
        self.vertex = vertex
        self.track = track
        ecal.insert(1, 'dph', 0.05 * ecal['ph'])  # pulse height uncertainty was given by instructor: 5%
        self.ecal = ecal
        v = np.max(vertex.dx/vertex.x, vertex.dy/vertex.y, vertex.dz/vertex.z, vertex.dphi/vertex.phi)
        t = np.max(track.dk/track.k, track.dtan_theta/track.tan_theta)
        e = np.max(ecal.dx/ecal.x, ecal.dy/ecal.y, ecal.dz/ecal.z)
        self.max_rel = np.max([v, t, e])

    def __repr__(self):
        return f'{self.particle} | ' \
               f'p={self.momentum} GeV | ' \
               f'v={len(self.vertex)} | ' \
               f't={len(self.track)} | ' \
               f'e={len(self.ecal)}'


class InjectionsCurator:

    def __init__(self, alpha: float,particle: str, momentum: float, data: list[Injection]):
        self.alpha = alpha
        self.particle = particle
        self.momentum = momentum, 0.01 * momentum   # momentum uncertainty was given by instructor: 1%
        self.data = data
