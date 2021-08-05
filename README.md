<img src="assets/aalto_logo_large.png" alt="Aalto University Logo" width="200"/>

# Projection Predictive Method for Bayesian Model Selection in Retail Time Series Forecasting

## Instructions (Technical)
### Preparing data and execution environment
1. Install `requirements.txt` and remember to activate the environment where the packages 
were installed before starting the jupyter kernel. It is recommended to start the kernel at the root of this project (in folder ``bsc-thesis``) by running
    ```commandline 
    jupyter notebook $NOTEBOOK_NAME
    ```
2. Load the dataset from [Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) 
(Corporación Favorita Grocery Sales Forecasting Competition)
3. Unzip the files into the ``data`` folder


### Using MCMC sampling
Consider the notes that [Facebook Prophet](https://facebook.github.io/prophet/docs/uncertainty_intervals.html)
has documented about upstream issues with PyStan

> There are upstream issues in PyStan for Windows which make MCMC sampling extremely slow. 
> The best choice for MCMC sampling in Windows is to use R, or Python in a Linux VM.