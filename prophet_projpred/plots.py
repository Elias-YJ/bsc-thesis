import matplotlib.pyplot as plt
import pandas as pd


def plot_submodels(m, submodels, future):
    """Create a plot for overviewing submodel and model performance over the
    train and test sets.

    :param m: ReferenceModel
    :param submodels: dict(projections, prediction_df, statistics)
    :param future: pd.DataFrame
    :return:
    """
    pred_fig = plt.figure(facecolor='w', figsize=(14, 6))
    pred_ax = pred_fig.add_subplot(111)

    for size, model_dict in submodels.items():
        proj_prediction = model_dict['prediction_df']
        pred_ax.plot(proj_prediction['ds'].values,
                     proj_prediction['yhat'],
                     label=size,
                     ls='-')
    pred_ax.plot(future['ds'].values, future['y'], 'r.', label="Test data")
    pred_ax.plot(future['ds'].values,
                 m.predictive_samples_mean(future)['yhat'],
                 label="Reference model",
                 ls='--')
    pred_ax.legend()
    pred_ax.set_ylabel('y')
    pred_ax.set_xlabel('ds')
    pred_ax.set_xlim([pd.to_datetime('2017-03-01'), future['ds'].tail(1)])
    plt.show()


def plot_submodel_trends(m, submodels, future):
    """Create a plot for overviewing submodel and model performance over the
    train and test sets.

    :param m: ReferenceModel
    :param submodels: dict(projections, prediction_df, statistics)
    :param future: pd.DataFrame
    :return:
    """
    pred_fig = plt.figure(facecolor='w', figsize=(14, 6))
    pred_ax = pred_fig.add_subplot(111)

    for size, model_dict in submodels.items():
        proj_prediction = model_dict['prediction_df']
        pred_ax.plot(proj_prediction['ds'].values,
                     proj_prediction['trend'],
                     label=size,
                     ls='-')
    pred_ax.plot(future['ds'].values,
                 m.predictive_samples_mean(future)['trend'],
                 label="Reference model",
                 ls='--')
    pred_ax.legend()
    pred_ax.set_ylabel('y')
    pred_ax.set_xlabel('ds')
    plt.show()


def plot_submodel_statistics(submodels):
    statistics_table = pd.DataFrame(columns=[
        'elpd', 'elpd_test', 'mape', 'mape_test', 'kl'])
    for size, model_dict in submodels.items():
        stat_row = pd.DataFrame(model_dict['statistics'], index=[size])
        statistics_table = statistics_table.append(
            stat_row, ignore_index=True)

    fig, ax = plt.subplots(5, 1, figsize=(13, 18))
    ax[0].plot(statistics_table['elpd'], label="elpd")
    ax[1].plot(statistics_table['elpd_test'], label="elpd test")
    ax[2].plot(statistics_table['mape'], label="mape", c='k')
    ax[3].plot(statistics_table['mape_test'], label="mape test")
    ax[4].plot(statistics_table['kl'], label="kl", c='r')
    for axis in ax:
        axis.legend(prop={'size': 16})
        axis.set_xlabel('variables')
    plt.show()

