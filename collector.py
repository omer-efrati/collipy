from collider import Collider


class DataCollector:

    def __init__(self, user: str, password: str, alpha: float):
        self.collider = Collider(user, password, alpha)
        self.raw_data = []
        self.data = []

    def collect(self, particle: str, momentum: float, n: int, threshold=0.25):
        data = []
        while len(self.data) != self.n:
            injections = self.collider.inject(self.particle, self.momentum, self.n)
            for inj in injections:
                if inj.max_rel < threshold:
                    self.data.append(inj)
