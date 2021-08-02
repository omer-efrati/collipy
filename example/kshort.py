"""
K-short data collection and analysis
"""
import pickle
from pathlib import Path
import collipy as cp


@cp.function_timer
def collect():
    alpha = 10
    particle = 'k-short'
    momentum = 3.7
    n = 5
    mode = cp.DecayMode(1, 2, 0)
    threshold = 0.2
    ac = cp.Accelerator(alpha)

    def cond(inj):
        """verify pi-plus and pi-minus (you will be surprised from some weird results)"""
        if inj.track.k.iloc[0] * inj.track.k.iloc[1] < 0:
            return True
        else:
            return False

    data = ac.collect(particle, momentum, n, mode, threshold, cond)
    data.dump(True)
    return data


if __name__ == '__main__':
    path = Path('data/kshort.pickle')
    folder = 'data'
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = collect()
