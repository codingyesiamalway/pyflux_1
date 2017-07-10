# implemented method in article
# https://datacrushblog.wordpress.com/2016/12/20/statistical-arbitrage-trading-pairs-in-python-using-correlation-cointegration-and-the-engle-granger-approach/

import sys
sys.path.append('./trading_pairs')
sys.path.append('./trading_pairs/technicaltools')
from technicaltools import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#from matplotlib.finance import candlestick
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
import seaborn as sns
import time
import random
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller as adf
import quandl
from sklearn import linear_model
from statsmodels.formula.api import ols

# ==============================================================================
# ==============================================================================
### load data
aapl_df = quandl.get("WIKI/AAPL")['Adj. Close'].to_frame('aapl')
goog_df = quandl.get("WIKI/GOOG")['Adj. Close'].to_frame('goog')
data_df = pd.merge(goog_df, aapl_df, left_index=True, right_index=True)

# ==============================================================================
# ==============================================================================
# run linear regression of aapl against goog. run test for different window. This shows that
# a spurious regression is obtained
for size in [800, 700, 500, 400, 300, 200, 100, 50, 20]:
    formula = 'aapl ~ goog'
    results = ols(formula, data_df.iloc[size:], hasconst=True).fit()
    resid= results.resid
    #data_df[['aapl_resid']].plot()
    hypotheses = 'goog = 0'
    t_test = results.t_test(hypotheses)
    print 'size: {}, tvalue: {}'.format(size, t_test.tvalue[0,0])

# run regression with all data and plot residual
results = ols(formula, data_df.iloc, hasconst=True).fit()
resid= results.resid
data_df['aapl_resid'] = resid
data_df['aapl_resid'].plot()

# ==============================================================================
#                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------
# Intercept     61.4154      2.828     21.713      0.000        55.863    66.968
# goog           0.0727      0.004     17.751      0.000         0.065     0.081
# ==============================================================================


# ==============================================================================
# ==============================================================================
# Check cointegration assuming cointegrating vector is know.
# Method from Hamilton p.582.
#1. check aapl and goog are both I(1) with drift. Expect to pass the check.
### a. use PhillipsPerron test
### b. use ADF test
#2. check that a'y is stationary. i.e. aapl - 0.0727 * goog is stationary with mean 61.4154.


# PhillipsPerron test on goog.
# for GOOG:
# MA(10) is used to fit the data
# stats: -0.287115074837, pvalue: 0.927309502487
# stats: -0.582869706432, pvalue: 0.920441499758
# h0: rho == 1 is rejected. There is no unit root.

# for AAPL
# MA(15) fits model the best.
# stats: 3.04749309416, pvalue: 1.0
# stats: 5.14080783224, pvalue: 0.999999999468
# h0: rho == 1 is not rejected. There is unit root.

test_series = goog_df
from arch.unitroot import PhillipsPerron

# fit MA on resid.
import statsmodels.tsa.arima_model as arma
for ma_lag in [7,8,9,10,11,12,13,15,20]:
    model = arma.ARMA(test_series, (0, ma_lag)).fit()
    print 'lag: {}, aic: {}'.format(ma_lag, model.aic)
pp = PhillipsPerron(test_series, trend='c', lags=10, test_type='tau')
print 'stats: {}, pvalue: {}'.format(pp.stat, pp.pvalue)
pp = PhillipsPerron(test_series, trend='c', lags=10, test_type='rho')
print 'stats: {}, pvalue: {}'.format(pp.stat, pp.pvalue)


### using adf test
### goog and aapl t-value
# 0.876758903592
# 0.999070159449

test_series = goog_df
from arch.unitroot import ADF
adf = ADF(goog_df, lags=24)
print adf.pvalue
adf = ADF(aapl_df, lags=24)
print adf.pvalue



### checking if residual is stationary
resid_series = data_df['aapl_resid'].dropna()
adf = ADF(resid_series, trend='nc', method='AIC', max_lags=20)
print adf.pvalue

pp = PhillipsPerron(resid_series, trend='nc', lags=10, test_type='tau')
print 'stats: {}, pvalue: {}'.format(pp.stat, pp.pvalue)
pp = PhillipsPerron(test_series, trend='c', lags=10, test_type='rho')
print 'stats: {}, pvalue: {}'.format(pp.stat, pp.pvalue)


# adf p-value: 0.103331915666
# thus, no cointegration is found.


# ==============================================================================
# ==============================================================================
# Check cointegration assuming cointegrating vector is unknown.
# Method from Hamilton p.582.