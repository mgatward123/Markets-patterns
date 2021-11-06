 
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

#import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
#from sklearn.externals import joblib
import ta

#import pyrenko
import scipy.stats as st
import scipy.optimize as opt
from sklearn.utils import resample
import datetime as dt
from dateutil.relativedelta import relativedelta


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

    ################
    print(x)

    #data = pdr.get_data_yahoo(x, interval = "1d", start="1990-07-15", end="2020-08-01") 
    
    data = pdr.get_data_yahoo(x, interval = "5d", start="2018-10-15", end="2020-09-01") 


    #data = data[:-330]

    #data = data[:-78]


    del data['Adj Close']

    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    price = data['Close'].tail(250)

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

    BD = current_pat[1] - current_pat[3]
    CE = current_pat[4] - current_pat[2]
    DF = current_pat[5] - current_pat[3]

    BD_range = np.array([1.12 - err_allowed, 1.618 + err_allowed])
    CE_range = np.array([1.618 - err_allowed, 2.24 + err_allowed])
    DF_range = np.array([0.40 - err_allowed, 0.60 + err_allowed])

    # BULL HAS TO BE GOING WITH UPTEND
    if AB<0 and BC>0 and CD < 0 and DE > 0 and EF < 0:
        if BD_range[0] < abs(BD) < BD_range[1] and CE_range[0] < abs(CE) < CE_range[1] and DF_range[0] < abs(DF) < DF_range[1]: 
            print('BULL')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # BEAR HAS TO BE GOING WITH UPTEND
    if AB>0 and BC<0 and CD > 0 and DE < 0 and EF > 0:
        if BD_range[0] < abs(BD) < BD_range[1] and CE_range[0] < abs(CE) < CE_range[1] and DF_range[0] < abs(DF) < DF_range[1]: 
            print('BEAR')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()



    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-4:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]
    #DE = current_pat[4] - current_pat[3]

    # Bullish
    if AB<0 and BC>0 and CD<0:
        #print('Bullish')
        BC_range = np.array([0.618 - err_allowed, 0.786 + err_allowed])*abs(AB)
        CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])*abs(BC)

        if BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    # Bearish
    if AB>0 and BC<0 and CD>0:
        #print('Bearish')
        BC_range = np.array([0.618 - err_allowed, 0.786 + err_allowed])*abs(AB)
        CD_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])*abs(BC)

        if BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1]:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    price = data['Close'].tail(500)

    #print(price)

    #price = data['Close'].tail(120)

    max_idx = list(argrelextrema(price.values, np.greater, order=5)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=5)[0]) 

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


    if AB > 0 and BC < 0 and CD > 0 and DE < 0 and EF > 0 and FG < 0:
        if A < B and C < B < D and D > B and D > C and D > E  and D > F and G < E < F and F > E and F > G and G > A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    if AB < 0 and BC > 0 and CD < 0 and DE > 0 and EF < 0 and FG > 0:
        if A > B and C > B > D and D < B and D < C and D < E  and D < F and G > E > F and F < E and F < G and G < A:
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()




    price = data['Close'].tail(30)

    #print(price)

    #price = data['Close'].tail(120)

    max_idx = list(argrelextrema(price.values, np.greater, order=1)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=1)[0]) 

    idx = max_idx

    idx.sort()

    current_idx = idx[-3:] + [len(price.values) - 1] 

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]


    HighLocationPrice = current_pat[1]



    AB_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])
    BC_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])

    # BEAR HAS TO BE GOING WITH UPTEND
    if AB<0 and BC<0 and -0.5 < CD < 0.5:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1]: 
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    # BULL HAS TO BE GOING WITH DOWNTREND
    if AB>0 and BC>0 and -0.5 < CD < 0.5:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1]: 
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()





    price = data['Close'].tail(30)

    #print(price)

    #price = data['Close'].tail(120)

    max_idx = list(argrelextrema(price.values, np.greater, order=1)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=1)[0]) 

    idx = min_idx

    idx.sort()

    current_idx = idx[-3:] + [len(price.values) - 1] 

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    AB = current_pat[1] - current_pat[0]
    BC = current_pat[2] - current_pat[1]
    CD = current_pat[3] - current_pat[2]


    HighLocationPrice = current_pat[1]


    AB_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])
    BC_range = np.array([1.27 - err_allowed, 1.618 + err_allowed])

    # BEAR HAS TO BE GOING WITH UPTEND
    if AB<0 and BC<0 and -0.2 < CD < 0.2:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1]: 
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    # BULL HAS TO BE GOING WITH DOWNTREND
    if AB>0 and BC>0 and -0.2 < CD < 0.2:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1]: 
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()



    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    price = data['Close'].tail(250)

    price = price[:-1]
    



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


    # BULLISH NEUTRAL TRAINGLE
    if AB>0 and BC<0 and CD>0 and DE<0:
        if A<B and A>C and A<E and A<D and B > A and B > C and B > E and D > B and C < A and C < B and C < D and C < E and D > B and D > E and E > A and E > C:
            print('Neutral-triangles Bullish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    # BEARISH NEUTRAL TRAINGLE
    if AB<0 and BC>0 and CD<0 and DE>0:
        if A>B and A<C and A>E and A>D and B < A and B < C and B < E and D < B and C > A and C > B and C > D and C > E and D < B and D < E and E < A and E < C:
            print('Neutral-triangles Bearish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # EXTRACTING TRAINGLES
    if AB<0 and BC>0 and CD<0 and DE>0:
        if A>B and A < C and A > E and A > D and B > D and B < A and B < C and B < E and C > A and C > E and C > D and C > B and D < A and D < B and D < C and D < E and E < C and E > D and E > B:
            print('EXTRACTING TRAINGLES BULLISH')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    if AB>0 and BC<0 and CD>0 and DE<0:
        if A<B and A > C and A < E and A < D and B < D and B > A and B > C and B > E and C < A and C < E and C < D and C < B and D > A and D > B and D > C and D > E and E > C and E < D and E < B:
            print('EXTRACTING TRAINGLES BEARISH')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # EXTRACTING 3rd EXTENSION FORMULA
    if AB<0 and BC>0 and CD<0 and DE>0:
        if A>B and A < C and A > D and B < C and B < D and C > E and C > D and C > B and C > A and D < E and D < C and D < A and  E < C and E > D:
            print('EXTRACTING TRAINGLES Bullish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # EXTRACTING 3rd EXTENSION FORMULA
    if AB>0 and BC<0 and CD>0 and DE<0:
        if A<B and A > C and A < D and B > C and B > D and C < E and C < D and C < B and C < A and D > E and D > C and D > A and  E > C and E < D:
            print('EXTRACTING TRAINGLES Bearish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    # 5th failure scanner 
    if AB>0 and BC<0 and CD>0 and DE<0:    
        if A<B and A>D and A<C and B<A and B<C and B<D and C>A and C>B and C>D and C>E and D>B and D<A and D<C and D<E and E>D and E<C and E<A:
            print('Fith failure scanner Bearish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()


    if AB<0 and BC>0 and CD<0 and DE>0:    
        if A>B and A<D and A>C and B>A and B>C and B>D and C<A and C<B and C<D and C<E and D<B and D>A and D>C and D>E and E<D and E>C and E>A:
            print('Fith failure scanner Bullish')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()




    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    #price = data['Close'].tail(250)

    #price = price[:-1]
    


    ####################### 4 WAVE ELLIOT

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


    # BULLISH DIAMETRIC
    if AB<0 and BC>0 and CD<0 and DE>0 and EF < 0 and FG>0:
        if A>C and A > B and A > D and A > F and C > B and C > D and C > F and C < E and B < D and B < F and D < A and D < C and D < E and F < D and F > B and F < G and F < C:
            print('BULLISH DIAMETRIC')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()

    # BEARISH DIAMETRIC
    if AB>0 and BC<0 and CD>0 and DE<0 and EF>0 and FG<0:
        if A<C and A < B and A < D and A < F and C < B and C < D and C < F and C > E and B > D and B > F and D > A and D > C and D > E and F > D and F < B and F > G and F > C:
            print('BEARISH DIAMETRIC')
            plt.plot(price.values)
            plt.scatter(current_idx, current_pat, c='r')
            plt.show()    


    ####################### 4 WAVE ELLIOT

    max_idx = list(argrelextrema(price.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=10)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    #current_idx = idx[-7:]

    current_pat = price.values[current_idx]


    start = min(current_idx)
    end = max(current_idx)

    print(current_pat)

    








