
#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr
from scipy.signal import argrelextrema
from collections import defaultdict
from datetime import timedelta

yf.pdr_override() 

import math  
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import statistics
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans

import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
from sklearn.externals import joblib
import ta

import pyrenko
import scipy.stats as st
import scipy.optimize as opt
from sklearn.utils import resample
import datetime as dt
from dateutil.relativedelta import relativedelta
import trendln


from feature_functions import *
from harmonic_functions import *

#import xgboost
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (12,8)
from sklearn import  metrics, model_selection
#from xgboost.sklearn import XGBClassifier

# DO THE REST OF JAN HAVE TO DELETE ROW

import pandas as pd

from pricelevels.cluster import ZigZagClusterLevels
from pricelevels.visualization.levels_with_zigzag import plot_with_pivots


ticker = [


'EURUSD=X',
'GBPUSD=X', 
'USDJPY=X',
'AUDUSD=X',
'USDCAD=X',
'EURGBP=X',
'EURJPY=X',
'GBPEUR=X',
'USDCHF=X',
'EURCHF=X',
'GBPJPY=X',
'EURCAD=X',
'CADJPY=X',
'GBPCAD=X',
'CADCHF=X',
'GBPCHF=X',
'CHFJPY=X',
'USDSGD=X',
'EURSGD=X',
'GBPSGD=X',
'SGDJPY=X',
'NZDUSD=X',
'GBPAUD=X',
'AUDJPY=X',
'AUDCAD=X',
'EURAUD=X',
'AUDNZD=X',
'NZDJPY=X',
'EURNZD=X',
'GBPNZD=X',
'AUDCHF=X',
'NZDCAD=X',
'NZDCHF=X',
'AUDGBP=X',
'NZDGBP=X',
'AUDEUR=X',
'NZDEUR=X',
'NZDAUD=X',
'AUDSGD=X',


]



longs = []
shorts = []


# Oct 07 2005 – Apr 03 2020

err_allowed = 50/100



for x in ticker:

    ################
    print(x)


    data = pdr.get_data_yahoo(x, interval = "5m", period = "1mo") 

    data = data[:-450]


    del data['Adj Close']

    print(data)

    print(price) 

    #price = price[:-1]
    ####################### 4 WAVE ELLIOT

    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-5:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]
    DE = current_pat[4] - current_pat[3]

    A = current_pat[0]
    B = current_pat[1]
    C = current_pat[2]
    D = current_pat[3]
    E = current_pat[4]


    # Bullish
    if AB>0 and BC<0 and CD>0 and DE<0:

        if A<B and B>C and C<D and D>E and E>B and C>A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # Bearish
    if AB<0 and BC>0 and CD<0 and DE>0 and EF<0:

        if A>B and B<C and C>D and D<E and E<B and C<A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    ####################### 5 WAVE ELLIOT

    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-6:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]
    DE = current_pat[4] - current_pat[3]
    EF = current_pat[5] - current_pat[4]
    #FG = current_pat[6] - current_pat[5]



    A = current_pat[0]
    B = current_pat[1]
    C = current_pat[2]
    D = current_pat[3]
    E = current_pat[4]
    F = current_pat[5]

    # Bullish
    if AB>0 and BC<0 and CD>0 and DE<0 and EF>0 and CD > AB:

        if A<B and B>C and C<D and D>E and E<F and E>B and C>A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # Bearish
    if AB<0 and BC>0 and CD<0 and DE>0 and EF<0 and CD < AB:

        if A>B and B<C and C>D and D<E and E>F and E<B and C<A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    ####################### a WAVE ELLIOT

    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-7:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]
    DE = current_pat[4] - current_pat[3]
    EF = current_pat[5] - current_pat[4]
    FG = current_pat[6] - current_pat[5]



    A = current_pat[0]
    B = current_pat[1]
    C = current_pat[2]
    D = current_pat[3]
    E = current_pat[4]
    F = current_pat[5]
    G = current_pat[6]

    # Bullish
    if AB>0 and BC<0 and CD>0 and DE<0 and EF>0 and CD > AB and FG<0:

        if A<B and B>C and C<D and D>E and E<F and E>B and C>A and G<E and G>C:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # Bearish
    if AB<0 and BC>0 and CD<0 and DE>0 and EF<0 and CD < AB and FG>0:

        if A>B and B<C and C>D and D<E and E>F and E<B and C<A and G>E and G<C:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    ####################### b WAVE ELLIOT


    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-8:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]
    DE = current_pat[4] - current_pat[3]
    EF = current_pat[5] - current_pat[4]
    FG = current_pat[6] - current_pat[5]
    GH = current_pat[7] - current_pat[6]



    A = current_pat[0]
    B = current_pat[1]
    C = current_pat[2]
    D = current_pat[3]
    E = current_pat[4]
    F = current_pat[5]
    G = current_pat[6]
    H = current_pat[7]


    # Bullish
    if AB>0 and BC<0 and CD>0 and DE<0 and EF>0 and CD > AB and FG<0 and GH>0:

        if A<B and B>C and C<D and D>E and E<F and E>B and C>A and G<E and G>C and H>G:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # Bearish
    if AB<0 and BC>0 and CD<0 and DE>0 and EF<0 and CD < AB and FG>0 and GH<0:

        if A>B and B<C and C>D and D<E and E>F and E<B and C<A and G>E and G<C and H<G:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()
