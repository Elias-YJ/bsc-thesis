import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def plot_submodels(m, submodels, future):
    """Create a plot for overviewing submodel and model performance over the
    train and test sets.

    :param m: ReferenceModel
    :param submodels: dict(projections, prediction_df, statistics)
    :param future: pd.DataFrame
    :return:
    """
    pred_fig = plt.figure(facecolor='w', figsize=(12, 6), dpi=100)
    pred_ax = pred_fig.add_subplot(111)

    for size, model_dict in submodels.items():
        proj_prediction = model_dict['prediction_df']
        pred_ax.plot(proj_prediction['ds'].values,
                     proj_prediction['yhat'],
                     label=size,
                     ls='-')

    test_indices = future.index.values > m.history.index.max()
    train_indices = future.index.values <= m.history.index.max()

    pred_ax.plot(future.loc[train_indices, 'ds'].values,
                 future.loc[train_indices, 'y'], 'k.', label="Training data")
    pred_ax.plot(future.loc[test_indices, 'ds'].values,
                 future.loc[test_indices, 'y'], 'r*', label="Test data")
    pred_ax.plot(future['ds'].values,
                 m.predictive_samples_mean(future)['yhat'],
                 label="Reference model",
                 ls='--')
    pred_ax.legend(loc='lower left')
    pred_ax.set_ylabel('log(total sales)')
    pred_ax.set_xlabel('date')
    pred_ax.set_xlim([pd.to_datetime('2017-03-01'), future['ds'].tail(1)])
    plt.show()


def plot_submodel_component(m, submodels, future, component='yhat', title=""):
    """Create a plot for overviewing submodel and model performance over the
    train and test sets.

    :param m: ReferenceModel
    :param submodels: dict(projections, prediction_df, statistics)
    :param future: pd.DataFrame
    :param component: str
    :return:
    """
    pred_fig = plt.figure(facecolor='w', figsize=(8, 4), dpi=120)
    pred_ax = pred_fig.add_subplot(111)

    for size, model_dict in submodels.items():
        proj_prediction = model_dict['prediction_df']
        pred_ax.plot(proj_prediction['ds'].values,
                     proj_prediction[component],
                     label=size,
                     ls='-')
    pred_ax.plot(future['ds'].values,
                 m.predictive_samples_mean(future)[component],
                 label="Reference model",
                 ls='--')
    pred_ax.axvline(pd.to_datetime('2017-05-1'), color='k', ls='--',
                    label='Beginning of test set ')
    pred_ax.legend(loc='upper left')
    pred_ax.set_ylabel('log(total sales)')
    pred_ax.set_xlabel('date')
    pred_ax.set_title(title)
    plt.show()


def plot_submodel_statistics(submodels):
    statistics_table = pd.DataFrame(columns=[
        'elpd', 'elpd_test', 'mape', 'mape_test', 'kl'])
    for size, model_dict in submodels.items():
        stat_row = pd.DataFrame(model_dict['statistics'], index=[size])
        statistics_table = statistics_table.append(
            stat_row, ignore_index=True)

    fig, ax = plt.subplots(5, 1, figsize=(8, 18), dpi=120)
    ax[0].plot(statistics_table['elpd'], label="elpd")
    ax[1].plot(statistics_table['elpd_test_30'], label="elpd test 1-30")
    ax[1].plot(statistics_table['elpd_test_60'], label="elpd test 31-60")
    ax[2].plot(statistics_table['mape'], label="mape", c='k')
    ax[3].plot(statistics_table['mape_test_30'], label="mape test 1-30")
    ax[3].plot(statistics_table['mape_test_60'], label="mape test 31-60")
    ax[4].plot(statistics_table['kl'], label="kl", c='r')
    for axis in ax:
        axis.legend(prop={'size': 16})
        axis.set_xlabel('variables')
    plt.show()


def plot_submodel_training_stats(submodels):
    statistics_table = pd.DataFrame(columns=[
        'elpd', 'elpd_test', 'mape', 'mape_test', 'kl'])
    for size, model_dict in submodels.items():
        stat_row = pd.DataFrame(model_dict['statistics'], index=[size])
        statistics_table = statistics_table.append(
            stat_row, ignore_index=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
    ax[0].plot(statistics_table['elpd'], label="ELPD")
    ax[1].plot(statistics_table['mape'], label="MAPE", c='k')
    ax[1].ticklabel_format(style='sci', scilimits=(-1, 1))
    #ax[0].set_title('Performance statistics over the training data')
    for axis in ax:
        axis.legend(prop={'size': 16})
        axis.set_xlabel('Submodel size')
    plt.show()


def plot_submodel_test_stats(submodels):
    statistics_table = pd.DataFrame(columns=[
        'elpd', 'elpd_test', 'mape', 'mape_test', 'kl'])
    for size, model_dict in submodels.items():
        stat_row = pd.DataFrame(model_dict['statistics'], index=[size])
        statistics_table = statistics_table.append(
            stat_row, ignore_index=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
    ax[0].plot(statistics_table['elpd_test_30'], 'b-', label="ELPD over days 1-30")
    ax[0].plot(statistics_table['elpd_test_60'], 'k-', label="ELPD over days 31-60")
    ax[1].plot(statistics_table['mape_test_30'], 'r--', label="MAPE over days 1-30")
    ax[1].plot(statistics_table['mape_test_60'], 'b--', label="MAPE over days 31-60")
    #ax[0].set_title('Performance statistics over the test data')
    for axis in ax:
        axis.legend(prop={'size': 16})
        axis.set_xlabel('Submodel size')
    plt.show()


def plot_model_distribution(m, submodel, future):
    """Create a plot for overviewing model distribution over the
    train and test sets.

    :param m: ReferenceModel
    :param submodel: list(projections, prediction_df, statistics)
    :param future: pd.DataFrame
    :return:
    """
    submodel = submodel.copy()
    fig, ax = plt.subplots(2, 1, facecolor='w', figsize=(12, 10), dpi=100)

    proj_prediction = submodel['prediction_df']
    ref_prediction = m.predict(future)
    ax[0].plot(proj_prediction['ds'].values,
               proj_prediction['yhat'],
               label='Mean prediction',
               ls='-',
               color='red')

    ax[0].fill_between(proj_prediction['ds'].values,
                       proj_prediction['yhat_lower'],
                       proj_prediction['yhat_upper'],
                       color='#910078', alpha=0.2,
                       label='Uncertainty interval')

    test_indices = future.index.values > m.history.index.max()
    train_indices = future.index.values <= m.history.index.max()

    ax[1].plot(future['ds'].values,
               ref_prediction['yhat'],
               label="Reference model",
               ls='-')

    ax[1].fill_between(proj_prediction['ds'].values,
                       ref_prediction['yhat_lower'],
                       ref_prediction['yhat_upper'],
                       color='#0072B2', alpha=0.2,
                       label='Uncertainty interval')
    for pred_ax in ax:
        pred_ax.plot(future.loc[train_indices, 'ds'].values,
                     future.loc[train_indices, 'y'], 'k.',
                     label="Data")
        pred_ax.plot(future.loc[test_indices, 'ds'].values,
                     future.loc[test_indices, 'y'], 'k.')
        pred_ax.axvline(pd.to_datetime('2017-05-01'), color='gray', ls='--',
                        label='Beginning of test set')
        pred_ax.legend(loc='lower left')
        pred_ax.set_ylabel('log(total sales)')
        pred_ax.set_xlabel('date')
        pred_ax.set_xlim([pd.to_datetime('2017-03-01'), future['ds'].tail(1)])
    plt.show()

