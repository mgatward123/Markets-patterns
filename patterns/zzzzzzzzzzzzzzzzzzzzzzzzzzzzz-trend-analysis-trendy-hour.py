
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

import trendy


for x in ticker:

        ################
    print(x)

    #hist = pdr.get_data_yahoo(x, interval = "1h",   period = "1mo")
    #hist = pdr.get_data_yahoo(x, interval = "1h",   period = "1mo")
    if x == 'AUDCAD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-12-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-25", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDUSD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDGBP=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-02", end="2020-04-28")

    if x == 'AUDNZD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'AUDEUR=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'AUDSGD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURAUD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURGBP=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-16", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-03", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")        

    if x == 'EURUSD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-09", end="2020-04-28")

    if x == 'EURCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-25", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-03", end="2020-04-28")

    if x == 'GBPAUD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-02", end="2020-04-28")

    if x == 'GBPUSD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPSGD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPEUR=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-08-15", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPNZD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPCAD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-30", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDAUD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDUSD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDCAD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDGBP=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-01", end="2020-04-28")

    if x == 'NZDCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'USDCAD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-04", end="2020-04-28")

    if x == 'USDCHF=X': 
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-05", end="2020-04-28")

    if x == 'USDSGD=X': 
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-01-01", end="2020-04-28")

    if x == 'USDJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-09", end="2020-04-28")    
    
    if x == 'CADJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")        

    if x == 'USDZAR=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-01-01", end="2020-04-28")

    if x == 'CADCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-10", end="2020-04-28")

    if x == 'EURCHF=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-23", end="2020-04-28")

    if x == 'EURSGD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")

    if x == 'GBPSGD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'CADJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURNZD=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-18", end="2020-04-28")        

    if x == 'NZDEUR=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-18", end="2020-04-28")

    if x == 'CHFJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")

    if x == 'SGDJPY=X':
        #df = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        df = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-23", end="2020-04-28")




    #df = pdr.get_data_yahoo(x, interval = "1m", start="2020-04-20", end="2020-04-21")

    df = df.reset_index()

    print(df)

    df = df['Close']

    #df = df[:-1]


    # Generate general support/resistance trendlines and show the chart
    # winow < 1 is considered a fraction of the length of the data set
    #trendy.gentrends(df, window = 1.0/3, charts = True)

    # Generate a series of support/resistance lines by segmenting the price history
    #trendy.segtrends(df, segments = 2, charts = True)  # equivalent to gentrends with window of 1/2
    trendy.segtrends(df, segments = 5, charts = True)  # plots several S/R lines

    # Generate smaller support/resistance trendlines to frame price over smaller periods
    #trendy.minitrends(df, window = 30, charts = True)

    # Iteratively generate trading signals based on maxima/minima in given window
    #trendy.iterlines(df, window = 30, charts = True)  # buy at green dots, sell at red dots  