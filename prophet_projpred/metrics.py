from scipy.stats import norm


def map_log_lik(ref_pred, yhat, submodel):
    # Standard deviation of estimates
    sigma_obs = submodel.params['sigma_obs'][0, 0]

    # Scaling of parameters
    y_scale = submodel.y_scale

    # Returns the log likelihood of observing target values equal to ref_pred
    log_lik = norm(loc=yhat, scale=sigma_obs*y_scale).logpdf(ref_pred).sum()
    return log_lik
