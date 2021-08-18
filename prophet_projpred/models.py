from prophet import Prophet
from .family import Gaussian
from .util import thinning
from . import stats
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

    def projection_model(self, regressors):
        proj = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                       holidays=self.holidays)
        regressors = [variable for variable in self.variables if
                      variable['name'] in regressors]

        for season in self.seasons:
            proj.add_seasonality(**season)

        for reg in regressors:
            proj.add_regressor(**reg)

        return proj

    def project(self, train: pd.DataFrame, future: pd.DataFrame,
                regressors, ndraws_pred=1):
        """Project parameter draws to submodel space that is
        spanned by the given regressors. Uses mean values from reference space
        if only one draw is selected.

        :param train: DataFrame
        :param future: DataFrame
        :param regressors: list
        :param ndraws_pred: int
        :return: DataFrame, Prophet
        """
        try:
            self.fit(train)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass

        if ndraws_pred > 1:
            ref_pred_draws = self.predictive_samples(future)
        else:
            ref_pred_draws = self.predictive_samples_mean(future)

        draw_indices = thinning(ndraws_pred, self.uncertainty_samples)

        # Project by fitting the submodel to reference model predictions
        projections = pd.DataFrame(index=future.index)
        for i in draw_indices:
            # Replace original data with reference model projections
            y = ref_pred_draws['yhat'].iloc[:, i]
            submodel_train = train.copy()
            submodel_train['y'] = y

            # Initialize submodel and fit to reference model predictions
            submodel = self.projection_model(regressors)
            submodel.fit(submodel_train)

            projection = submodel.predict(future).loc[:, ['yhat']]\
                .rename(columns={'yhat': i})
            projections = pd.concat([projections, projection],
                                    ignore_index=True, axis=1)
        return projections, submodel

    def predictive_samples(self, df):
        samples = super().predictive_samples(df)
        output = {}
        for key in samples:
            output[key] = pd.DataFrame(data=samples[key])
        return output

    def predictive_samples_mean(self, future):
        prediction = self.predict(future).loc[:, ['yhat']]
        return {'yhat': prediction}

    def sample_model(self, df, seasonal_features, iteration, s_a, s_m):
        """Generate predictive samples from posterior draws.
        Parameters
        ----------
        df: Prediction dataframe.
        seasonal_features: pd.DataFrame of seasonal features.
        iteration: Int sampling iteration to use parameters from.
        s_a: Indicator vector for additive components
        s_m: Indicator vector for multiplicative components
        Returns
        -------
        Dataframe with trend and yhat, each like df['t'].
        """
        trend = self.sample_predictive_trend(df, iteration)

        beta = self.params['beta'][iteration]
        Xb_a = np.matmul(seasonal_features.values,
                         beta * s_a.values) * self.y_scale
        Xb_m = np.matmul(seasonal_features.values, beta * s_m.values)

        sigma_obs = self.params['sigma_obs'][iteration]

        return pd.DataFrame({
            'yhat': trend * (1 + Xb_m) + Xb_a,
            'trend': trend,
            'sigma_obs': sigma_obs
        })

    def sample_posterior_predictive(self, df):
        """Prophet posterior predictive samples.
        Parameters
        ----------
        df: Prediction dataframe.
        Returns
        -------
        Dictionary with posterior predictive samples for the forecast yhat and
        for the trend component.
        """
        n_iterations = self.params['k'].shape[0]
        samp_per_iter = max(1, int(np.ceil(
            self.uncertainty_samples / float(n_iterations)
        )))

        # Generate seasonality features once so we can re-use them.
        seasonal_features, _, component_cols, _ = (
            self.make_all_seasonality_features(df)
        )

        sim_values = {'yhat': [], 'trend': [], 'sigma_obs': []}
        for i in range(n_iterations):
            for _j in range(samp_per_iter):
                sim = self.sample_model(
                    df=df,
                    seasonal_features=seasonal_features,
                    iteration=i,
                    s_a=component_cols['additive_terms'],
                    s_m=component_cols['multiplicative_terms'],
                )
                for key in sim_values:
                    sim_values[key].append(sim[key])
        for k, v in sim_values.items():
            sim_values[k] = np.column_stack(v)
        return sim_values

    def search(self, train: pd.DataFrame, future: pd.DataFrame,
               ndraws_search=1, ndraws_pred=20):
        predictions = self.predictive_samples(future)
        yhat = predictions['yhat'].iloc[:, 0:ndraws_pred]
        sigma_obs = predictions['sigma_obs'].iloc[:, 0:ndraws_pred]
        y = np.broadcast_to(
            future['y'].values.reshape((len(future['y']), 1)),
            yhat.shape
        )

        added_variables = []
        path = {}
        while len(added_variables) < len(self.variables):
            kl_table, variables = self.search_step(
                train, future, added_variables, ndraws_search)
            print(kl_table)
            # Add the variable which increases KL divergence the least
            added_variables.append(kl_table.loc[0, 'variable'])

            # Project draws from the reference to the previously selected
            # submodel space and calculate mean predictions
            proj_predictions, _ = self.project(
                train, future, variables, ndraws_pred=ndraws_pred)

            # Calculate projection statistics
            test_indices = future.index.values > self.history.index.max()
            mape = stats.mape(y, proj_predictions, test_indices=test_indices)

            dis = self.family.dispersion(
                yhat.values, sigma_obs.values, proj_predictions.values)
            dis = np.broadcast_to(dis, yhat.shape)
            loglik = self.family.loglik(y, proj_predictions, dis)
            elpd = stats.elpd(loglik, test_indices=test_indices)

            path[len(added_variables)] = {
                'variables': variables,
                'predictions': proj_predictions,
                'mape': mape,
                'elpd': elpd
            }
        return path

    def search_step(self, train: pd.DataFrame, future: pd.DataFrame,
                    initial_variables, ndraws_search=1):
        """Increase submodel variables by the best option and return results
        """
        try:
            ref_fit = self.fit(train)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass
        ref_pred = self.predictive_samples_mean(future)

        variables = [variable['name'] for variable in self.variables
                     if variable['name'] not in initial_variables]
        kl_divs = []
        for var in variables:
            prediction, submodel = self.project(
                train, future, initial_variables + [var], ndraws_search)
            kl = self.family.kl(ref_pred['yhat'].values, prediction.values)
            kl_divs.append(kl)
        result = pd.DataFrame(
            {'variable': variables, 'kl': kl_divs}
        ).sort_values(by='kl', ignore_index=True)
        best = result.loc[0, 'variable']
        return result, initial_variables + [best]
