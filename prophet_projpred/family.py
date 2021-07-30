from abc import ABC, abstractmethod
import numpy as np


class Family(ABC):
    @abstractmethod
    def kl(self, ref, sub):
        pass


class Gaussian(Family):
    def kl(self, ref, sub, weights=1):
        return np.sum(weights * (sub - ref)**2)
