"""
Accelerator
===========

"""
from . import ssh
from collections import Counter
from ..injection import DecayMode, Injection, InjectionCollection
from timeit import default_timer as timer
from ..helper import progress_bar


class Accelerator:
    """
    Accelerator have the ability to inject particles and collect data
    
    Parameters
    ----------
    username : str
        Your TAU username
    password : str
        Your TAU password
    alpha : float
        magnetic field multiplicity factor: 'alpha' * B_0
        alpha is in the range [0.1, 10]

    """

    def __init__(self, username: str, password: str, alpha=1):
        self.alpha = alpha
        self._geant = ssh.GSH(username, password, alpha)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        low = 0.1
        high = 10
        if not low <= value <= high:
            raise ValueError(f"alpha must meet the condition: {low} <= alpha <= {high}")
        self._alpha = value

    def _inject(self, particle: str, momentum: float, n: int) -> [Injection]:
        """Injecting `n` particles and returns a list of Injection objects for successful injections
        therefore the length of the list is `n` at most (may be less)"""
        lst = self._geant.inject(particle, momentum, n)
        res = []
        for tmp in lst:
            res.append(Injection(tmp[0], tmp[1], tmp[2]))
        return res

    def collect(self, particle, momentum, n, mode, threshold):
        """
        Collecting injections that meet the threshold and decay mode

        Parameters
        ----------
        particle : str
            particle name as stated at the README.md file
        momentum : float
            particle's momentum
        n : int
            collecting at least this number of injections that meet the conditions
        mode : DecayMode
            reflects the decay mode to of the required injections to collect
        threshold : float
            max relative error for parameters in desired injection

        Returns
        -------
            data : InjectionCollection

        """
        start = timer()
        data = []
        cnt = Counter()
        # Arbitrary lower bound for number of injections per loop
        times = n if 100 < n else 100
        while len(data) < n:
            lst = self._geant.inject(particle, momentum, times)
            # counts the total number of events
            cnt['total'] += len(lst)
            for vertex, track, ecal in lst:
                progress_bar(len(data[:n]), n, start)
                injection = Injection(vertex, track, ecal)
                # counts the number of events with certain decay mode
                cnt[injection.mode.tup] += 1
                if mode <= injection.mode:  # 'injection.mode' contains 'mode'
                    if injection.max_rel <= threshold:
                        data.append(injection)
            # updating number of injections per loop for higher efficiency
            times = int((n - len(data)) / (len(data) / cnt['total'])) if 0 < len(data) else times
        print()
        return data, cnt
