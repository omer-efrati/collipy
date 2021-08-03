"""
Accelerator
===========

"""
from distutils.util import strtobool
from pathlib import Path
from collections import Counter
import concurrent
from time import perf_counter as timer
import numpy as np
from paramiko.ssh_exception import AuthenticationException
from . import ssh
from ..injection import DecayMode, Injection, InjectionCollection
from ..helper import progress_bar


class Accelerator:
    """
    Accelerator have the ability to inject particles and collect data
    
    Parameters
    ----------
    alpha : float
        magnetic field multiplicity factor: 'alpha' * B_0
        alpha is in the range [0.1, 10]

    Notes
    -----
    This class will request your TAU credentials and log it IN YOUR COMPUTER ONLY
    you can refuse and login each time you run your module

    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.creds_path = Path('credentials.conf')
        if self.creds_path.is_file():
            with open(self.creds_path, 'r') as creds:
                username = creds.readline().strip('\n')
                password = creds.readline().strip('\n')
        else:
            username = input('Username: ')
            password = input('Password: ')
            while True:
                try:
                    save_creds = strtobool(input('Would you like to save your credentials? y/n'))
                    break
                except ValueError:
                    print('Invalid command, please try again...')
            if save_creds:
                with open(self.creds_path, 'w') as creds:
                    creds.write(f'{username}\n')
                    creds.write(f'{password}\n')
        try:
            self._geant = ssh.GSH(username, password, alpha)
        except AuthenticationException as e:
            if self.creds_path.is_file():
                # delete credentials file if authentication failed
                self.creds_path.unlink()
            raise e

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

    def collect(self, particle, momentum, n, mode, threshold, cond=lambda inj: True):
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
        cond : (inj: Injection) -> bool
            function that returns True if the injection is sufficient, False otherwise

        Returns
        -------
            data : InjectionCollection

        """
        start = timer()
        data = []
        cnt = Counter()
        upper_bound = 0
        # `times` empirically chosen to be a good injections/loop ratio
        times = 40
        while len(data) < n:
            lst = self._geant.inject(particle, momentum, times)
            # counts the total number of events
            cnt['total'] += len(lst)
            for vertex, track, ecal in lst:
                injection = Injection(vertex, track, ecal)
                # counts the number of events with certain decay mode
                cnt[injection.mode.tup] += 1
                if mode <= injection.mode:  # 'injection.mode' contains 'mode'
                    if injection.max_rel <= threshold:
                        if cond(injection):
                            data.append(injection)
                            upper_bound = injection.max_rel if upper_bound < injection.max_rel else upper_bound
                progress_bar(len(data[:n]), n, start, f'{len(data)} injections collected')
        print()
        data = InjectionCollection(self.alpha, particle, momentum, data, mode, upper_bound, cnt)
        return data

    def acollect(self, particles, modes, momenta, n, threshold, cond=lambda inj: True):
        """
        running `collect` function concurrently for different particles and/or momenta

        Parameters
        ----------
        particles
            list of particles names to inject
        modes
            modes corresponding to the particles list
        momenta
            momentum list for each particle
        n
            number of injections per momentum required
        threshold
            maximum relative error allowed
        cond : (inj: Injection) -> bool
            function that returns True if the injection is sufficient, False otherwise

        Returns
        -------
            list[InjectionCollection] with all data
        """

        def run(particle, momentum, mode):
            return self.collect(particle, momentum, n, mode, threshold, cond)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(run,
                                   np.repeat(particles, len(momenta)),
                                   np.tile(momenta, len(particles)),
                                   np.repeat(modes, len(momenta)))
        data = {key: [] for key in particles}
        for ic in results:
            data[ic.particle].append(ic)
        return data
