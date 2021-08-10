from prophet import Prophet
from .family import Gaussian
import logging
import pandas as pd
import numpy as np

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

    def ref_pred(self, future, ndraws_pred):
        predictions = self.predictive_samples(future)['yhat'][:, 0:ndraws_pred]
        predictions = pd.DataFrame(predictions)
        predictions['ds'] = future['ds']
        predictions = predictions.melt(id_vars=['ds'], var_name='draw',
                                       value_name='yhat')
        return predictions

    def ref_mean_pred(self, future):
        prediction = self.predict(future).loc[:, ['ds', 'yhat']]
        prediction['draw'] = 0
        return prediction

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

    def project(self, future, regressors, ndraws_pred=1):
        """Project parameter draws to submodel space that is
        spanned by the given regressors. Uses mean values from reference space
        if only one draw is selected.

        :param future: DataFrame
        :param regressors: list
        :param ndraws_pred: int
        :return: DataFrame, Prophet
        """
        future = future.copy()
        try:
            self.fit(future)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass

        if ndraws_pred > 1:
            ref_pred_draws = self.ref_pred(future, ndraws_pred=ndraws_pred)
        else:
            ref_pred_draws = self.ref_mean_pred(future)

        # Project by fitting the submodel to reference model predictions
        predictions = pd.DataFrame()
        for i in range(ndraws_pred):
            y = ref_pred_draws.groupby(
                'draw').get_group(i).loc[:, ['yhat']].set_index(future.index)
            future['y'] = y
            submodel = self.projection_model(regressors)
            submodel.fit(future)
            projection = submodel.predict(future).loc[:, ['ds', 'yhat']]
            projection['draw'] = i
            predictions = predictions.append(projection)
        return predictions, submodel

    def search(self, future: pd.DataFrame):
        added_variables = []
        path = {}
        while len(added_variables) < len(self.variables):
            kl_table, variables = self.search_step(future, added_variables)
            print(kl_table)
            # Add the variable which increases KL divergence the least
            added_variables.append(kl_table.loc[0, 'variable'])

            # Project draws from the reference to the previously selected
            # submodel space and calculate mean predictions
            predictions, _ = self.project(future, variables, ndraws_pred=10)
            mean_predictions = predictions.groupby(
                by='ds').aggregate('mean')
            pred_indices = future.index.values > self.history.index.max()

            # Calculate projection statistics
            y_test = future.loc[pred_indices, 'y'].values
            yhat_test = mean_predictions.loc[pred_indices, 'yhat'].values
            mape = np.mean(np.abs((y_test - yhat_test)/y_test))

            dis = self.family.dispersion()
            elpd = self.family.loglik()

            path[len(added_variables)] = {
                'variables': variables,
                'predictions': predictions,
                'mape': mape
            }
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
        for var in variables:
            prediction, submodel = self.project(
                future, initial_variables + [var])
            kl = self.family.kl(ref_pred['yhat'], prediction['yhat'])
            kl_divs.append(kl)
        result = pd.DataFrame(
            {'variable': variables, 'kl': kl_divs}
        ).sort_values(by='kl', ignore_index=True)
        best = result.loc[0, 'variable']
        return result, initial_variables + [best]
