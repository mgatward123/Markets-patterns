
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




'MMM',
'ABT',
'ABBV',
'ABMD',
'ACN',
'ATVI',
'ADBE',
'AMD',
'AAP',
'AFL',
'A',
'APD',
'AKAM',
'ALK',
'ALB',
'ARE',
'ALXN',
'ALGN',
'ALLE',
'AGN',
'ADS',
'LNT',
'ALL',
'GOOGL',
'GOOG',
'MO',
'AMZN',
#'AMCR',
'AEE',
'AAL',
'AEP',
'AXP',
'AIG',
'AMT',
'AWK',
'AMP',
'ABC',
'AME',
'AMGN',
'APH',
'ADI',
'ANSS',
'ANTM',
'AON',
'AOS',
'APA',
'AIV',
'AAPL',
'AMAT',
'APTV',
'ADM',
#'ARNC',
'ANET',
'AJG',
'AIZ',
'ATO',
'T',
'ADSK',
'ADP',
'AZO',
'AVB',
'AVY',
#'BKR',
'BLL',
'BAC',
'BK',
'BAX',
'BDX',
'BBY',
'BIIB',
'BLK',
'BA',
'BKNG',
'BWA',
'BXP',
'BSX',
'BMY',
'AVGO',
'BR',
'CHRW',
'COG',
'CDNS',
'CPB',
'COF',
'CPRI',
'CAH',
'KMX',
'CCL',
'CAT',
'CBOE',
'CBRE',
'CDW',
'CE',
'CNC',
'CNP',
'CTL',
'CERN',
'CF',
'SCHW',
'CHTR',
'CVX',
'CMG',
'CB',
'CHD',
'CI',
'XEC',
'CINF',
'CTAS',
'CSCO',
'C',
'CFG',
'CTXS',
'CLX',
'CME',
'CMS',
'KO',
'CTSH',
'CL',
'CMCSA',
'CMA',
'CAG',
'CXO',
'COP',
'ED',
'STZ',
'COO',
'CPRT',
'GLW',
#'CTVA',
'COST',
'COTY',
'CCI',
'CSX',
'CMI', 
'CVS',
'DHI',
'DHR',
'DRI',
'DVA',
'DE',
'DAL',
'XRAY',
'DVN',
'FANG',
'DLR',
'DFS',
'DISCA',
'DISCK',
'DISH',
'DG',
'DLTR',
'D',
'DOV',
#'DOW',	
'DTE',	
'DUK',		
'DRE',		
'DD',	
'DXC',		
'ETFC',	
'EMN',	
'ETN',	
'EBAY',	
'ECL',
'EIX',	
'EW',	
'EA',		
'EMR',	
'ETR',	
'EOG',
'EFX',	
'EQIX',
'EQR',
'ESS',
'EL',		
'EVRG',	
'EXC',	
'EXPE',		
'EXPD',	
'EXR',
'XOM',	
'FFIV',	
'FB',	
'FAST',	
'FRT',	
'FDX',	
'FIS',		
'FITB',	
'FE',
'FRC',	
'FISV',
'FLT',
'FLIR',		
'FLS',		
'FMC',	
'F',	
'FTNT',
'FTV',
'FBHS',		
#'FOXA',	
#'FOX',	
'BEN',	
'FCX',	
'GPS',	
'GRMN',	
'IT',	
'GD',	
'GE',
'GIS',
'GM',	
'GPC',	
'GILD',	
#'GL',	
'GPN',	
'GS',	
'GWW',	
'HRB',		
'HAL',
'HBI',
'HOG',	
'HIG',	
'HAS',		
'HCA',	
#'PEAK',		
'HP',
'HSIC',		
'HSY',	
'HES',	
'HPE',		
'HLT',
'HFC',		
'HOLX',		
'HD',	
'HON',	
'HRL',		
'HST',	
'HPQ',
'HUM',	
'HBAN',	
'HII',	
'IEX',	
'IDXX',		
'INFO',		
'ITW',
'ILMN',		
'IR',	
'INTC',	
'ICE',	
'IBM',	
'INCY',		
'IP',
'IPG',	
'IFF',		
'INTU',	
'ISRG',
'IVZ',
'IPGP',		
'IQV',	
'IRM',		
'JKHY',		
#'J',		
'JBHT',	
'SJM',	
'JNJ',
'JCI',		
'JPM',		
'JNPR',	
'KSU',
'K',
'KEY',	
'KEYS',	
'KMB',	
'KIM',		
'KMI',		
'KLAC',	
'KSS',
'KHC',		
'KR',
'LB',	
'LHX',
'LH',	
'LRCX',		
'LW',	
'LVS',	
'LEG',
'LDOS',	
'LEN',		
#'LLY',	
'LNC',	
'LIN',		
'LYV',	
'LKQ',	
'LMT',	
'L',	
'LOW',	
'LYB',		
'M',
'MRO',	
'MPC',
'MKTX',	
'MAR',		
'MLM',		
'MAS',	
'MA',	
'MKC',	
'MXIM',	
'MCD',
'MCK',	
'MDT',	
'MRK',	
'MET',	
'MTD',		
'MGM',	
'MCHP',
'MU',
'MSFT',
'MAA',
'MHK',		
'TAP',
'MDLZ',
'MNST',	
'MCO',
'MS',	
'MOS',	
'MSI',	
'MSCI',		
'MYL',
'NDAQ',	
'NOV',
'NTAP',		
'NFLX',
'NWL',	
'NEM',	
'NWSA',		
'NWS',
'NEE',	
'NLSN',	
'NKE',
'NI',	 
'NBL',
'JWN',	
'NSC',
'NTRS',	
'NOC',
#'NLOK',	
'NCLH',	
'NRG', 
#'NUE',	
'NVDA',	
'NVR',	
'ORLY',
'OXY',
'ODFL',
'OMC',
'OKE',	
'ORCL',
'PCAR',	
'PKG',
'PH',
'PAYX',	
'PYPL',	
'PNR',	
'PBCT',	
'PEP',	 
'PKI',	
'PRGO',
'PFE',	
'PM',	
'PSX',	
'PNW',	
'PXD',	
'PNC',	
'PPG',	
'PPL',
'PFG',
'PG',	
'PGR',	
'PLD',		
'PRU',	
'PEG',	
'PSA',
'PHM',	
'PVH',		
'QRVO',	
'PWR',
'QCOM',	
'DGX',
'RL',	
'RJF',	
'RTN',
'O',	
'REG',		
'REGN',	
'RF',	
'RSG',	
'RMD',	
'RHI',	
'ROK',	
'ROL',
'ROP',	
'ROST',	
'RCL',
'SPGI',	
'CRM',
'SBAC',	
'SLB',	
'STX',	
'SEE',	
'SRE',	
'NOW',	
'SHW',	
'SPG',	
'SWKS',	
'SLG',
'SNA',	
'SO',	
'LUV',	
'SWK',	
'SBUX',	
'STT',	
'STE',	
'SYK',	
'SIVB',	
'SYF',	
'SNPS',		
'SYY',
'TMUS',		
'TROW',	
'TTWO',	
'TPR',
'TGT',
'TEL',	
'FTI',	
'TFX',
'TXN',	
'TXT',	
'TMO',	
'TIF',		
'TJX',	
'TSCO',
'TDG',	
'TRV',	
#'TFC',	
'TWTR',	
'TSN',	
'UDR',	
'ULTA',	
'USB',
'UAA',			
'UA',	
'UNP',	
'UAL',	
'UNH',	
'UPS',	
'URI',		
'UTX',	
'UHS',	
'UNM',	
'VFC',	
'VLO',
'VAR',	
'VTR',
'VRSN',	
'VRSK',
'VZ',
#'VRTX',	
#'VIAC',	
'V',
'VNO',	 
'VMC',	
'WRB',	
'WAB',	
'WMT',	
'WBA',	
'DIS',	
'WM',	
'WAT',	
'WEC',	
#'WCG',	
'WFC',	
'WELL',
'WDC',
'WU',	
'W',
'WY',		
'WHR',	
'WMB',	
'WLTW',	
'WYNN',	
'XEL',
'XRX',	
'XLNX',
'XYL',
'YUM',	
'ZBRA',	
'ZBH',	
'ZION',	
'ZTS',




]



longs = []
shorts = []


# Oct 07 2005 – Apr 03 2020

err_allowed = 50/100



for x in ticker:
    print(x)

    data =  pdr.get_data_yahoo(x, interval = "1d", start="2000-03-02", end="2019-07-06") 


    del data['Adj Close']

    print(data)

    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    price = data['Close'].tail(150)

    print(price)

    #price = price[:-1]

    #price = data['Close'].tail(120)


    for i in range(145, len(price)):
 
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

