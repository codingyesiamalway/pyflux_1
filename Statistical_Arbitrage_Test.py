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
#2. check that a'y is stationary. i.e. aapl - 0.0727 * goog is stationary with mean 61.4154.