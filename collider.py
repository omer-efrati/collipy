from server import GSH
import pandas as pd
from particles import Particle


class Collider:

    def __init__(self, user: str, password: str, alpha=1):
        low = 0.1
        high = 10
        if not low <= alpha <= high:
            raise ValueError(f"alpha must meet the condition: {low} <= alpha <= {high}")
        self.alpha = alpha
        self.geant = GSH(user, password, alpha)

    def inject(self, particle: Particle, momentum: float, n: int) -> pd.DataFrame:
        txt = self.geant.inject(particle.name, momentum, n)
        df = self.geant.parse(txt)
        return df
