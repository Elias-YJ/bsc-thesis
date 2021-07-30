from .models import ReferenceModel
from .metrics import map_log_lik
from prophet.diagnostics import cross_validation
import pandas as pd


def varsel(reference_model: ReferenceModel, future: pd.DataFrame,
           proj_samples=0):
    added_variables = []
    validations = []
    while len(added_variables) < len(reference_model.variables)-1:
        result, cv = simple_varsel(reference_model, future, added_variables,
                                   proj_samples)
        added_variables.append(result.loc[0, 'variable'])
        validations.append(cv)
        print(result)
    return validations


def simple_varsel(reference_model: ReferenceModel, future: pd.DataFrame,
                  initial_variables, proj_samples=0):
    """Select the best single-variable submodel and return all results
    """
    try:
        ref_fit = reference_model.fit(future)
    except Exception:
        # Reference model is fitted. Proceed to projecting
        pass
    ref_pred = reference_model.predict(future)['yhat']
    ref_lp = reference_model.stan_backend.stan_fit.extract('lp__')['lp__']

    variables = [variable['name'] for variable in reference_model.variables
                 if variable['name'] not in initial_variables]
    kl_divs = []
    submodels = {}
    for var in variables:
        prediction, submodel = reference_model.project(
            future, initial_variables+[var], proj_samples=proj_samples)
        if proj_samples == 0:
            sub_lp = map_log_lik(ref_pred, prediction['yhat'], submodel)
        else:
            sub_lp = submodel.stan_backend.stan_fit.extract('lp__')['lp__']
        kl = 1/len(ref_lp)*sum(ref_lp-sub_lp)
        kl_divs.append(kl)
        submodels[var] = submodel
        print(var)
    result = pd.DataFrame(
        {'variable': variables, 'kl': kl_divs}
    ).sort_values(by='kl', ignore_index=True)
    best = result.loc[0, 'variable']
    cv = cross_validation(submodels[best], initial='365.25 days',
                          period='60 days', horizon='30 days')
    return result, cv
