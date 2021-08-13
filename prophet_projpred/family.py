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
        return np.sum(np.sum(weights * (sub - ref)**2))

    def dispersion(self, ref, sigma_obs, sub, weights=1):
        if weights == 1:
            weights = np.broadcast_to(np.ones(1), ref.shape)
        dis = np.sqrt(np.sum(weights/np.sum(weights)*(sigma_obs + (ref - sub)**2), axis=0))
        return dis

    def loglik(self, y, yhat, dis):
        return np.mean(norm.logpdf(y, loc=yhat, scale=dis), axis=0)
