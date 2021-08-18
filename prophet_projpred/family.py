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

    @abstractmethod
    def interval(self, yhat, dis, alpha):
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
        """Calculates the log likelihood matrix
        Let N be the number of data points and S the number of draws

        :param y: NxS matrix where the original Nx1 data is duplicated S times
        :param yhat: NxS matrix of yhat posterior draws
        :param dis: NxS matrix where the original 1xS dispersion parameters are
        duplicated for each data point
        :return: NxS log likelohood matrix
        """
        return norm.logpdf(y, loc=yhat, scale=dis)

    def interval(self, yhat, dis, alpha):
        """Return the upper and lower bounds of the confidence interval at
        confidence level alpha

        :param yhat: Nx1 vector
        :param dis: float
        :param alpha: float, within [0,1]
        :return:
        """
        return norm.interval(alpha, loc=yhat, scale=dis)
