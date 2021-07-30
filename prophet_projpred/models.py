from prophet import Prophet
from .family import Gaussian
import logging
import pandas as pd

logger = logging.getLogger('prophet')
logger.setLevel(logging.WARNING)


class ReferenceModel(Prophet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables = []
        self.seasons = []
        self.family = Gaussian()

    def add_regressor(self, name, prior_scale=None,
                      standardize='auto', mode=None):
        self.variables.append(
            {'name': name,
             'prior_scale': prior_scale})
        super().add_regressor(name, prior_scale, standardize, mode)

    def add_seasonality(self, name, period, fourier_order,
                        prior_scale=None, **kwargs):
        self.seasons.append(
            {'name': name,
             'period': period,
             'fourier_order': fourier_order})
        super().add_seasonality(name, period, fourier_order, prior_scale,
                                **kwargs)

    def ref_pred(self, future, n_draws_pred):
        return self.predictive_samples(future)['yhat'][:, n_draws_pred]

    def ref_mean_pred(self, future):
        return self.predict(future)['yhat']

    def projection_model(self, regressors=[]):
        proj = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                       holidays=self.holidays)
        regressors = [variable for variable in self.variables if
                      variable['name'] in regressors]

        for season in self.seasons:
            proj.add_seasonality(**season)

        for reg in regressors:
            proj.add_regressor(**reg)

        return proj

    def project(self, future, regressors):
        """Project mean of parameter draws to submodel space that is
        spanned by the given regressors

        :param future: DataFrame
        :param regressors: list
        :return: DataFrame, Prophet
        """
        future = future.copy()
        submodel = self.projection_model(regressors)
        try:
            self.fit(future)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass

        # Project by fitting the submodel to reference model predictions
        future['y'] = self.ref_mean_pred(future)
        submodel.fit(future)
        projection = submodel.predict(future)
        return projection, submodel

    def search(self, future: pd.DataFrame):
        added_variables = []
        path = []
        while len(added_variables) < len(self.variables):
            result, step = self.search_step(future, added_variables)
            print(result)
            added_variables.append(result.loc[0, 'variable'])
            path.append(step)
        return path

    def search_step(self, future: pd.DataFrame,
                    initial_variables):
        """Increase submodel variables by the best option and return results
        """
        try:
            ref_fit = self.fit(future)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass
        ref_pred = self.ref_mean_pred(future)

        variables = [variable['name'] for variable in self.variables
                     if variable['name'] not in initial_variables]
        kl_divs = []
        submodels = {}
        for var in variables:
            prediction, submodel = self.project(
                future, initial_variables + [var])
            kl = self.family.kl(ref_pred, prediction['yhat'])
            kl_divs.append(kl)
            submodels[var] = submodel
        result = pd.DataFrame(
            {'variable': variables, 'kl': kl_divs}
        ).sort_values(by='kl', ignore_index=True)
        best = result.loc[0, 'variable']
        return result, initial_variables + [best]
