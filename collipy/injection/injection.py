import pickle
from pathlib import Path
from collections import Counter
import pandas as pd
from .properties import DecayMode, calc_max_rel


class Injection:
    """Injection information and properties (for convenience and readability)"""
    def __init__(self, vertex: pd.DataFrame, track: pd.DataFrame, ecal: pd.DataFrame):
        self.vertex = vertex
        self.track = track
        self.ecal = ecal
        self.mode = DecayMode(vertex, track, ecal)
        self.max_rel = calc_max_rel(vertex, track, ecal)

    def __repr__(self):
        return f'{self.mode} {self.max_rel:0.2g}'


class InjectionCollection:

    def __init__(self, alpha, particle, momentum, data, mode, threshold, cnt: Counter):
        """Container for injection's data that meet the following requirements:
           * particle's decay mode follows 'mode' decay
           * maximum relative uncertainty for each parameter calculated is smaller than 'threshold'
        Parameters
        ----------
        alpha : flaot
            multiplicity factor for the magnetic field
        particle : str
            particle's name (as stated in README.md)
        momentum : float
            particle's momentum
        data : list[Injection]
            list of injections
        mode : DecayMode
            decay mode that all the injections met
        threshold : float
            maximum relative uncertainty
        cnt : Counter
            counter for all modes encountered during injections
            pay attention: len(data) <= sum(cnt)
        """
        self.particle = particle
        self.momentum = momentum, 0.01 * momentum  # momentum uncertainty was given by instructor: 1%
        self.data = data
        self.alpha = alpha
        self.mode = mode
        self.threshold = threshold
        self.cnt = cnt

    def __repr__(self):
        return f'({self.particle}, {self.momentum[0]}, {len(self.data)})'

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not have to specify it.
            data = pickle.load(f)
        return data

    def get_df(self):
        """Create a DataFrame object representing `self` injections"""
        vframes, tframes, eframes = [], [], []
        for injection in self.data:
            vframes.append(injection.vertex)
            tframes.append(injection.track)
            eframes.append(injection.ecal)
        v = pd.concat(vframes, keys=pd.RangeIndex(len(vframes)))
        t = pd.concat(tframes, keys=pd.RangeIndex(len(tframes)))
        e = pd.concat(eframes, keys=pd.RangeIndex(len(eframes)))
        return v, t, e

    def dump(self, path=None):
        """
        Pickling `InjectionCollection` object

        Parameters
        ----------
        path : str, optional
            if path is not given, saves the file in same dir of your module

        """
        if path is None:
            path = Path(f'{self.particle}_{self.momentum[0]}.pickle')
        else:
            path = Path(path)
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
