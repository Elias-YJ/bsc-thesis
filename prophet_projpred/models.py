from prophet import Prophet
from .family import Gaussian
from .util import thinning
from . import stats
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger('prophet')
logger.setLevel(logging.WARNING)

logging.getLogger("pystan").propagate = False


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

    def compute_submodels(self, path, train, future,
                          ndraws_pred=20, iter=1e4):
        """Convenience function for creating the submodels that are suggested
        by the search.

        :param path: list
        :param train: pd.DataFrame
        :param future: pd.DataFrame
        :param ndraws_pred: int
        :param iter: int
        :return:
        """
        submodels = {}
        for var_list in path:
            projections, statistics = self.project(
                train, future, var_list, ndraws=ndraws_pred, iter=iter)
            yhat = projections['yhat'].mean(axis=1)
            trend = projections['trend'].mean(axis=1)
            trend_extra = (projections['trend'] +
                           projections['extra_add']).mean(axis=1)
            sigma = projections['sigma_obs'].loc[0, :].mean()
            yhat_lower, yhat_upper = self.family.interval(
                yhat, sigma, self.interval_width)

            proj_prediction = pd.DataFrame(
                data={'yhat': yhat,
                      'yhat_upper': yhat_upper,
                      'yhat_lower': yhat_lower,
                      'trend': trend,
                      'trend_extra': trend_extra,
                      'ds': future['ds'].values},
                index=future.index
            )

            submodels[len(var_list)] = {
                'projections': projections,
                'prediction_df': proj_prediction,
                'statistics':  statistics
            }

        return submodels

    def init_submodel(self, regressors):
        proj = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                       changepoint_range=self.changepoint_range,
                       holidays=self.holidays,
                       n_changepoints=self.n_changepoints)
        regressors = [variable for variable in self.variables if
                      variable['name'] in regressors]

        for season in self.seasons:
            proj.add_seasonality(**season)

        for reg in regressors:
            proj.add_regressor(**reg)

        return proj

    def project(self, train: pd.DataFrame, future: pd.DataFrame,
                regressors, ndraws=1, iter=1e4):
        """Project parameter draws to submodel space that is
        spanned by the given regressors. Uses mean values from reference space
        if only one draw is selected.

        :param train: DataFrame
        :param future: DataFrame
        :param regressors: list
        :param ndraws: int
        :param iter: int
        :return: DataFrame, Prophet
        """
        try:
            self.fit(train)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass

        draw_indices = thinning(ndraws, self.uncertainty_samples)
        if ndraws > 1:
            ref_pred_draws = self.predictive_samples(future)
            ref_yhat = ref_pred_draws['yhat'].iloc[:, draw_indices]
            y = np.broadcast_to(
                future['y'].values.reshape((len(future['y']), 1)),
                ref_yhat.shape
            )
            sigma_obs = ref_pred_draws['sigma_obs'].iloc[
                        :, draw_indices].values
        else:
            ref_pred_draws = self.predictive_samples_mean(future)
            ref_yhat = ref_pred_draws['yhat'].iloc[:, 0:ndraws]
            sigma_obs = self.stan_fit.extract()['sigma_obs'][0:ndraws]
            y = future['y'].values

        # Project by fitting the submodel to reference model predictions
        projections = {'yhat': pd.DataFrame(index=future.index),
                       'trend': pd.DataFrame(index=future.index),
                       'extra_add': pd.DataFrame(index=future.index)}
        for i in range(len(ref_yhat.columns.values)):
            # Replace original data with reference model projections
            y_sub = ref_yhat.iloc[:, i]
            submodel_train = train.copy()
            submodel_train['y'] = y_sub

            # Initialize submodel and fit to reference model predictions
            submodel = self.init_submodel(regressors)
            submodel.fit(submodel_train, iter=iter)

            projection = submodel.predict(future)
            yhat = projection.loc[:, ['yhat']].rename(columns={'yhat': i})
            trend = projection.loc[:, ['trend']].rename(columns={'trend': i})

            projections['yhat'] = pd.concat(
                [projections['yhat'], yhat], ignore_index=True, axis=1)
            projections['trend'] = pd.concat(
                [projections['trend'], trend], ignore_index=True, axis=1)
            try:
                extra_add = projection.loc[:, ['extra_regressors_additive']]. \
                    rename(columns={'extra_regressors_additive': i})
                projections['extra_add'] = pd.concat(
                    [projections['extra_add'], extra_add],
                    ignore_index=True, axis=1)
            except KeyError:
                extra_add = pd.DataFrame(data={
                    i: np.zeros(len(future.index))
                }, index=future.index)
                projections['extra_add'] = pd.concat(
                    [projections['extra_add'], extra_add],
                    ignore_index=True, axis=1)
        dis = self.family.dispersion(
            ref_yhat.values, sigma_obs, projections['yhat'].values)
        dis = np.broadcast_to(dis, ref_yhat.shape)
        projections['sigma_obs'] = pd.DataFrame(data=dis, index=future.index)

        # Calculate projection statistics
        test_indices = future.index.values > self.history.index.max()
        test_indices30 = (future.index.values > self.history.index.max()) & (
                future.index.values + 30 < future.index.max()
        )
        test_indices60 = future.index.values > self.history.index.max() + 30
        train_indices = future.index.values <= self.history.index.max()
        loglik = self.family.loglik(y, projections['yhat'],
                                    projections['sigma_obs'])

        statistics = {'kl': self.family.kl(ref_yhat.values,
                                           projections['yhat'].values),
                      'elpd': stats.elpd(loglik, indices=train_indices),
                      'elpd_se': stats.elpd_se(loglik, indices=train_indices),
                      'elpd_test_30': stats.elpd(loglik,
                                              indices=test_indices30),
                      'elpd_test_60': stats.elpd(loglik,
                                                 indices=test_indices60),
                      'elpd_test_se': stats.elpd_se(loglik,
                                                    indices=test_indices),
                      'mape': stats.mape(y, projections['yhat'].values,
                                         indices=train_indices),
                      'mape_test_30': stats.mape(y, projections['yhat'].values,
                                              indices=test_indices30),
                    'mape_test_60': stats.mape(y, projections['yhat'].values,
                                   indices=test_indices60)}
        return projections, statistics

    def predictive_samples(self, df):
        samples = super().predictive_samples(df)
        output = {}
        for key in samples:
            output[key] = pd.DataFrame(data=samples[key])
        return output

    def predictive_samples_mean(self, future):
        prediction = self.predict(future)
        yhat = prediction.loc[:, ['yhat']]
        trend = prediction.loc[:, ['trend']]
        try:
            trend_extra = prediction.loc[:, 'trend'] +\
                          prediction.loc[:, 'extra_regressors_additive']
        except KeyError:
            trend_extra = prediction.loc[:, ['trend']]
        return {'yhat': yhat, 'trend': trend, 'trend_extra': trend_extra}

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
               ndraws_search=1):
        """Search which variables should a submodel contain for all possible
        submodel sizes.

        :param train:
        :param future:
        :param ndraws_search:
        :return:
        """

        path = [[]]
        while len(path[-1]) < len(self.variables):
            kl_table, variables = self.search_step(
                train, future, path[-1], ndraws_search)
            print(kl_table)
            # Add the variable which increases KL divergence the least
            path.append(variables)
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
            prediction, statistics = self.project(
                train, future, initial_variables + [var], ndraws_search,
                iter=1e4
            )
            kl_divs.append(statistics['kl'])
        result = pd.DataFrame(
            {'variable': variables, 'kl': kl_divs}
        ).sort_values(by='kl', ignore_index=True)
        best = result.loc[0, 'variable']
        return result, initial_variables + [best]
