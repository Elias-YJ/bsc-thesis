from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np


class Family(ABC):
    @abstractmethod
    def kl(self, ref, sub):
        pass

    @abstractmethod
    def dispersion(self, ref, refvar, sub):
        pass

    @abstractmethod
    def loglik(self, y, yhat, dis):
        pass


class Gaussian(Family):
    def kl(self, ref, sub, weights=1):
        return np.sum(weights * (sub - ref)**2)

    def dispersion(self, ref, sigma_obs, sub):
        return np.sqrt(sigma_obs + (ref - sub)**2)

    def loglik(self, y, yhat, dis):
        return np.sum(norm.logpdf(y, loc=yhat, scale=dis))
