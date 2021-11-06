
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


import plotly.graph_objects as go



ticker = [

#'AUDCAD=X',
#'AUDCHF=X',
#'AUDJPY=X',
#'EURAUD=X',
#'EURGBP=X',
#'EURJPY=X',
#'EURUSD=X',
#'EURCHF=X',
#'GBPUSD=X', 
#'GBPEUR=X',
#'GBPNZD=X',
#'GBPJPY=X',
#'GBPCAD=X',
#'GBPCHF=X',
#'NZDJPY=X',
#'NZDUSD=X',
#'USDCAD=X',
#'USDCHF=X', 
#'USDJPY=X',
#'CADJPY=X',
#'USDZAR=X',
#'CADCHF=X',


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


    data = pdr.get_data_yahoo(x, interval = "1d", start="2008-05-05", end="2019-12-08") 

    del data['Adj Close']

    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    df = data.tail(4)

    df = df.reset_index()

    #print(df)

    #price = price[:-1]

    dateFirst = df.iloc[[0]] 
    dateSecond = df.iloc[[1]] 
    dateThird = df.iloc[[2]] 
    dateFourth = df.iloc[[3]] 

    dateFirstOPEN = float(dateFirst['Open'])
    dateSecondOPEN = float(dateSecond['Open'])
    dateThirdOPEN = float(dateThird['Open'])
    dateFourthOPEN = float(dateFourth['Open'])

    dateFirstCLOSE = float(dateFirst['Close'])
    dateSecondCLOSE = float(dateSecond['Close'])
    dateThirdCLOSE = float(dateThird['Close'])
    dateFourthCLOSE = float(dateFourth['Close'])

    dateFirst = float(dateFirst['Close'])
    dateSecond = float(dateSecond['Close'])
    dateThird = float(dateThird['Close'])
    dateFourth = float(dateFourth['Close'])

    # BULLISH QUAD
    if dateFirstOPEN < dateFirstCLOSE and dateSecondOPEN < dateSecondCLOSE and dateThirdOPEN < dateThirdCLOSE and dateFourthOPEN < dateFourthCLOSE and dateSecond > dateFirst and dateThird < dateSecond and dateFourth < dateThird:
        print(x)
        print('BULLISH QUAD Success')

    # BEARISH QUAD
    if dateFirstOPEN > dateFirstCLOSE and dateSecondOPEN > dateSecondCLOSE and dateThirdOPEN > dateThirdCLOSE and dateFourthOPEN > dateFourthCLOSE and dateSecond < dateFirst and dateThird > dateSecond and dateFourth > dateThird:
        print(x)
        print('BEARISH QUAD Success')

    # BULLISH QUAD
    #if dateSecond > dateFirst and dateThird < dateSecond and dateFourth < dateThird:
    #    print(x)
    #    print('BULLISH QUAD Success')


    # BEARISH QUAD
    #if dateSecond < dateFirst and dateThird > dateSecond and dateFourth > dateThird:
    #    print(x)
    #    print('BEARISH QUAD Success')
