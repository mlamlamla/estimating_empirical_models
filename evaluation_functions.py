import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as ppt
from sklearn import linear_model
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from linearmodels import OLS
from linearmodels import PanelOLS
from linearmodels import PooledOLS
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

def plot_residuals_for_variance(exog,endog):
    # Perform PooledOLS
    mod = PooledOLS(endog, exog)
    pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
    # Store values for checking homoskedasticity graphically
    fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
    residuals_pooled_OLS = pooledOLS_res.resids
    # 3A. Homoskedasticity
    #import matplotlib.pyplot as plt
     # 3A.1 Residuals-Plot for growing Variance Detection
    fig, ax = ppt.subplots()
    ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
    ax.axhline(0, color = 'r', ls = '--')
    ax.set_xlabel('Predicted Values', fontsize = 15)
    ax.set_ylabel('Residuals', fontsize = 15)
    ax.set_title('Homoskedasticity Test', fontsize = 30)
    ppt.show()
    return residuals_pooled_OLS