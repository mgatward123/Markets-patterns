
#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr
from scipy.signal import argrelextrema


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
    print(x)

    data = pdr.get_data_yahoo(x, interval = "5m", period = "1mo") 

    data = data[:-450]


    del data['Adj Close']

    print(data)

    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    price = data['Close'].tail(300)

    print(price)


    #price = data['Close'].tail(120)


    for i in range(295, len(price)):
 
        max_idx = list(argrelextrema(price.values[:i], np.greater, order=10)[0])
        min_idx = list(argrelextrema(price.values[:i], np.less, order=10)[0])

        idx = max_idx + min_idx + [len(price.values[:i]) - 1]

        idx.sort()

        current_idx = idx[-5:]

        start = min(current_idx)
        end = max(current_idx)

        current_pat = price.values[current_idx]

        # GARTLY
        XA = current_pat[1] - current_pat[0]
        AB = current_pat[2] - current_pat[1]
        BC = current_pat[3] - current_pat[2]
        CD = current_pat[4] - current_pat[3]
        #DE = current_pat[5] - current_pat[4]

        AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed]) * abs(BC)

        if XA>0 and AB<0 and BC>0 and CD<0:

            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 

                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

        elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 
                
                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

        # BUTTERFLY 
        AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
        BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)



        if XA>0 and AB<0 and BC>0 and CD<0:


            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 

                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

        elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 
                
                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()


        # BAT
        AB_range = np.array([0.382 - err_allowed, 0.5 + err_allowed]) * abs(XA)
        BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)

        if XA>0 and AB<0 and BC>0 and CD<0:

            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 

                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

        elif XA < 0 and AB > 0 and BC < 0 and CD > 0:

            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 
                
                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()


        # CRAB
        AB_range = np.array([0.382 - err_allowed, 0.618 + err_allowed]) * abs(XA)
        BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
        CD_range = np.array([2.24 - err_allowed, 3.618 + err_allowed]) * abs(BC)

        if XA>0 and AB<0 and BC>0 and CD<0:
            
            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 

                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

        elif XA < 0 and AB > 0 and BC < 0 and CD > 0:


            if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]: 
                
                plt.plot(np.arange(start, i), price.values[start:i])
                plt.plot(current_idx, current_pat, c='r')
                plt.show()

