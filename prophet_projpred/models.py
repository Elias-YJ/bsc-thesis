from prophet import Prophet
import logging

logger = logging.getLogger('prophet')
logger.setLevel(logging.WARNING)


class ReferenceModel(Prophet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables = []
        self.seasons = []

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

    def projection_model(self, regressors=[], mcmc_samples=100):
        proj = Prophet(changepoint_prior_scale=0.01, holidays=self.holidays,
                       mcmc_samples=mcmc_samples)
        regressors = [variable for variable in self.variables if
                      variable['name'] in regressors]

        for season in self.seasons:
            proj.add_seasonality(**season)

        for reg in regressors:
            proj.add_regressor(**reg)

        return proj

    def project(self, future, regressors, proj_samples=0):
        future = future.copy()
        submodel = self.projection_model(regressors, mcmc_samples=proj_samples)
        try:
            self.fit(future)
        except Exception:
            # Reference model is fitted. Proceed to projecting
            pass
        ref_forecast = self.predict(future)

        # Project by fitting the submodel to reference model predictions
        future['y'] = ref_forecast['yhat']
        submodel.fit(future)
        projection = submodel.predict(future)
        return projection, submodel
