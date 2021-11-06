
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

    #hist = pdr.get_data_yahoo(x, interval = "1h",   period = "1mo")
    #hist = pdr.get_data_yahoo(x, interval = "1h",   period = "1mo")
    if x == 'AUDCAD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-12-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-25", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDUSD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-19", end="2020-04-28")

    if x == 'AUDGBP=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-02", end="2020-04-28")

    if x == 'AUDNZD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'AUDEUR=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'AUDSGD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURAUD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-02", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURGBP=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-16", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-03", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")        

    if x == 'EURUSD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-09", end="2020-04-28")

    if x == 'EURCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-25", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-03", end="2020-04-28")

    if x == 'GBPAUD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-02", end="2020-04-28")

    if x == 'GBPUSD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPSGD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-05", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPEUR=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-08-15", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPNZD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPCAD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-30", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'GBPCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-30", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDAUD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDUSD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDCAD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'NZDGBP=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-01", end="2020-04-28")

    if x == 'NZDCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'USDCAD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-04-04", end="2020-04-28")

    if x == 'USDCHF=X': 
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-05", end="2020-04-28")

    if x == 'USDSGD=X': 
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-01-01", end="2020-04-28")

    if x == 'USDJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-09", end="2020-04-28")    
    
    if x == 'CADJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")        

    if x == 'USDZAR=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-01-01", end="2020-04-28")

    if x == 'CADCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-10", end="2020-04-28")

    if x == 'EURCHF=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-23", end="2020-04-28")

    if x == 'EURSGD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")

    if x == 'GBPSGD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'CADJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-07-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-17", end="2020-04-28")

    if x == 'EURNZD=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-10-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-18", end="2020-04-28")        

    if x == 'NZDEUR=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-18", end="2020-04-28")

    if x == 'CHFJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-25", end="2020-04-28")

    if x == 'SGDJPY=X':
        #hist = pdr.get_data_yahoo(x, interval = "1h", start="2019-09-01", end="2019-11-28")
        hist = pdr.get_data_yahoo(x, interval = "1h", start="2020-03-23", end="2020-04-28")


    del hist['Adj Close']

    print(hist)

    #hist = hist[:-1]

    h = hist.Close.tolist()

    mins, maxs = trendln.calc_support_resistance(hist[-1000:].Close)
    minimaIdxs, pmin, mintrend, minwindows = trendln.calc_support_resistance((hist[-1000:].Low, None)) #support only
    mins, maxs = trendln.calc_support_resistance((hist[-1000:].Low, hist[-1000:].High))
    (minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows) = mins, maxs
    minimaIdxs, maximaIdxs = trendln.get_extrema(hist[-1000:].Close)
    maximaIdxs = trendln.get_extrema((None, hist[-1000:].High)) #maxima only
    minimaIdxs, maximaIdxs = trendln.get_extrema((hist[-1000:].Low, hist[-1000:].High))
    fig = trendln.plot_support_resistance(hist[-1000:].Close) # requires matplotlib - pip install matplotlib
    plt.savefig('suppres.svg', format='svg')
    plt.show()
    plt.clf() #clear figure
    #fig = trendln.plot_sup_res_date((hist[-1000:].Low, hist[-1000:].High), hist[-1000:].index) #requires pandas
    #plt.savefig('suppres.svg', format='svg')
    #plt.show()
    #plt.clf() #clear figure
    #curdir = '.'
    #trendln.plot_sup_res_learn(curdir, hist)  