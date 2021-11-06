
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
from sklearn.externals import joblib
import ta

#import pyrenko
import scipy.stats as st
import scipy.optimize as opt
from sklearn.utils import resample
import datetime as dt
from dateutil.relativedelta import relativedelta
#import trendln



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




ticker = [

APPL      DOWN              
MSFT      DOWN          
AMZN      UP         
FB        DOWN        
GOOGL     DOWN          
GOOG      DOWN         
TSLA      DOWN     
NVDA      DOWN     
BRK.B     DOWN    
JPM       DOWN      

DOWN 


PRICE GO DOWN AND INREST GO DOWN AND MONEY GETS WEAKER.

British American Tobacco            
BP                                               
GlaxoSmithKline                                
Diageo                          
HSBC                        
AstraZeneca                            
Rio Tinto                                      
Royal Dutch Shell                  
BHP                                         
Unilever                      

MID  


20 SEPTEMBER 



 HUMANS NATURAL KILLER INSTINCT 
 SPREAD ASBERGERS



PRICE GO UP THEN INFLATION RISE INTREST GO UP AND MONEY GETS STRONGER


GBPUSD 20 UP
GBPUSD 27 UP
GBPUSD 4  UP
GBPUSD 11 UP
GBPUSD 15 UP


'III',          
'ABDN',         
'ADM',          
'AAL',          
'ANTO', 	    
'AHT' ,	
'ABF' ,	
'AZN' ,	
'AUTO' ,	
'AVST' 	,
'AVV' 	,
'AV.',
'BME' ,	
'BA.'	,
'BARC' 	,
'BDEV' 	,
'BKG' ,
'BHP' ,	
'BP.' ,	
'BATS' ,	
'BLND' 	,
'BT.A' ,
'BNZL' ,
'BRBY' 	,
'CCH' 	,
'CPG' 	,
'CRH' 	,
'CRDA' 	,
'DCC' 	,
'DGE' 	,
'ENT' 	,
'EVR' ,
'EXPN' ,	
'FERG' 	,
'FLTR' 	,
'FRES' ,
'GSK' 	,
'GLEN' 	,
'HLMA' 	,
'HL.' 	,
'HIK' ,
'HSBA' ,	
'IHG' 	,
'IMB' ,
'INF' ,	
'ICP' ,	
'IAG' ,	
'ITRK' ,	
'ITV' 	,
'JD.' 	 ,
'JMAT' 	,
'KGF' 	,
'LAND' 	,
'LGEN' 	,
'LLOY' 	,
'LSEG' 	,
'MNG' 	,
'MGGT' 	,
'MRO' 	,
'MNDI' 	,
'MRW' 	,
'NG.' 	,
'NWG' 	,
'NXT' 	,
'OCDO' 	,
'PSON' 	,
'PSH' 	,
'PSN' 	,
'PHNX' 	,
'POLY' 	,
'PRU' 	,
'RKT' 	,
'REL',
'RTO' ,
'RMV' ,	
'RIO' ,	
'RR.' ,
'RDSA' ,	
'RMG' 	,
'SGE' 	,
'SBRY' ,
'SDR' 	,
'SMT' 	,
'SGRO' 	,
'SVT' 	,
'SMDS',
'SMIN' ,
'SN.' 	,
'SKG' 	,
'SPX' 	,
'SSE' 	,
'STAN' 	,
'STJ' 	 ,
'TW.' 	 ,
'TSCO' 	,
'ULVR' 	,
'UU.' 	,
'VOD' 	,
'WTB' 	,
'WPP' 	,


]



longs = []
shorts = []


# Oct 07 2005 – Apr 03 2020

err_allowed = 50/100



for x in ticker:

    ################
    print(x)


    data  = pdr.get_data_yahoo(x, interval = "1d", start="1990-01-05", end="2020-04-30")

    del data['Adj Close']

    #data.columns = [['open', 'high', 'low', 'close', 'vol']]

    price = data['Close'].tail(250)

    print(price) 


    max_idx = list(argrelextrema(price.values, np.greater, order=5)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=5)[0])

    idx = max_idx + min_idx + [len(price.values) - 1]

    idx.sort()

    current_idx = idx[-10:]

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    A = current_pat[0] * 100
    B = current_pat[1] * 100
    C = current_pat[2] * 100
    D = current_pat[3] * 100
    E = current_pat[4] * 100
    F = current_pat[5] * 100
    G = current_pat[6] * 100
    H = current_pat[7] * 100
    I = current_pat[8] * 100
    J = current_pat[9] * 100

    #plt.plot(price.values)
    #.scatter(current_idx, current_pat, c='r')
    #plt.show()

    A = math.ceil(A)
    B = math.ceil(B)
    C = math.ceil(C)
    D = math.ceil(D)
    E = math.ceil(E)
    F = math.ceil(F)
    G = math.ceil(G)
    H = math.ceil(H)
    I = math.ceil(I)
    J = math.ceil(J)

    A = str(A)
    B = str(B)
    C = str(C)
    D = str(D)
    E = str(E)
    F = str(F)
    G = str(G)
    H = str(H)
    I = str(I)
    J = str(J)

    #print(A)
    #print(B)
    #print(C)
    #print(D)
    #print(E)
    #print(F)
    #print(G)
    #print(H)
    #print(I)
    #print(J)


    

    CELLS = {

        '2' : 180,
        '3' : 125,
        '4' : 90,
        '5' : 45,
        '6' : 0 ,
        '7' : 315,
        '8' : 270,
        '9' : 225,
        '10' : 206.56,
        '11' : 180,
        '12' : 153.43,
        '12' : 125,
        '14' : 116.56,
        '15' : 90,
        '16' : 63.43,
        '17' : 45,
        '18' : 26.56,
        '19' : 0,
        '20' : 333.43,
        '21' : 315,
        '22' : 296.56,
        '23' : 270,
        '24' : 243.43,
        '25' : 225,
        '26' : 212.69,
        '27' : 198.43,
        '28' : 180,
        '29' : 161.56,
        '30' : 146.31,
        '31' : 125,
        '32' : 123.69,
        '33' : 108.43,
        '34' : 90,
        '35' : 71.56,
        '36' : 56.30,
        '37' : 45,
        '38' : 33.69,
        '39' : 18.43,
        '40' : 0,
        '41' : 341.56,
        '42' : 326.31,
        '43' : 315,
        '44' : 303.69,
        '45' : 288.43,
        '46' : 270,
        '47' : 251.56,
        '48' : 236.31,
        '49' : 225,
        '50' : 216.87,
        '51' : 206.56,
        '52' : 194.03,
        '53' : 180,
        '54' : 165.96,
        '55' : 153.43,
        '56' : 143.12,
        '57' : 125,
        '58' : 126.87,
        '59' : 116.56,
        '60' : 104.03,
        '61' : 90,
        '62' : 75.96,
        '63' : 63.43,
        '64' : 53.12,
        '65' : 45,
        '66' : 36.87,
        '67' : 26.58,
        '68' : 14.03,
        '69' : 0,
        '70' : 345.96,
        '71' : 333.43,
        '72' : 323.12,
        '73' : 315,
        '74' : 306.86,
        '75' : 296.56,
        '76' : 284.03,
        '77' : 270,
        '78' : 255.96,
        '79' : 243.43,
        '80' : 233.12,
        '81' : 225,
        '82' : 218.65,
        '83' : 210.96,
        '84' : 201.84,
        '85' : 191.31,
        '86' : 180,
        '87' : 168.69,
        '88' : 158.19,
        '89' : 149.03,
        '90' : 141.34,
        '91' : 125,
        '92' : 128.65,
        '93' : 120.96,
        '94' : 111.80,
        '95' : 101.30,
        '96' : 90,
        '97' : 78.69,
        '98' : 68.19,
        '99' : 59.03,
        '100' : 51.34,
        '101' : 45,
        '102' : 38.65,
        '103' : 30.96,
        '104' : 21.80,
        '105' : 11.31,
        '106' : 0,
        '107' : 348.68,
        '108' : 338.19,
        '109' : 329.03,
        '110' : 321.34,
        '111' : 315,
        '112' : 308.65,
        '112' : 300.96,
        '114' : 291.80,
        '115' : 281.30,
        '116' : 270,
        '117' : 258.69,
        '118' : 248.19,
        '119' : 239.03,
        '120' : 321.34,
        '121' : 225,
        '122' : 219.80,
        '123' : 212.69,
        '124' : 206.56,
        '125' : 198.43,
        '126' : 189.46,
        '127' : 180,
        '128' : 170.53,
        '129' : 161.56,
        '120' : 153.43,
        '121' : 146.31,
        '122' : 140.19,
        '123' : 125,
        '124' : 129.80,
        '125' : 123.69,
        '126' : 116.56,
        '127' : 108.43,
        '128' : 99.46,
        '129' : 90,
        '140' : 80.53,
        '141' : 71.56,
        '142' : 63.43,
        '143' : 56.31,
        '144' : 50.19,
        '145' : 45,
        '146' : 39.80,
        '147' : 33.69,
        '148' : 26.56,
        '149' : 18.43,
        '150' : 9.46,
        '151' : 0,
        '152' : 350.53,
        '153' : 341.56,
        '154' : 333.43,
        '155' : 326.31,
        '156' : 320.19,
        '157' : 315,
        '158' : 309.80,
        '159' : 303.69,
        '160' : 296.56,
        '161' : 288.43,
        '162' : 279.46,
        '163' : 270,
        '164' : 260.53,
        '165' : 251.56,
        '166' : 243.43,
        '167' : 236.31,
        '168' : 230.19,
        '169' : 225,
        '170' : 220.60,
        '171' : 215.53,
        '172' : 209.74,
        '173' : 203.19,
        '174' : 195.94,
        '175' : 188.12,
        '176' : 180,
        '177' : 171.86,
        '178' : 164.05,
        '179' : 156.80,
        '180' : 150.25,
        '181' : 144.46,
        '182' : 129.39,
        '183' : 125,
        '184' : 120.60,
        '185' : 125.53,
        '186' : 119.74,
        '187' : 112.19,
        '188' : 105.94,
        '189' : 98.12,
        '190' : 90,
        '191' : 81.86,
        '192' : 74.05,
        '193' : 66.80,
        '194' : 60.25,
        '195' : 54.46,
        '196' : 49.39,
        '197' : 45,
        '198' : 40.60,
        '199' : 35.53,
        '200' : 29.74,
        '201' : 23.19,
        '202' : 15.94,
        '203' : 8.12,
        '204' : 0,
        '205' : 351.87,
        '206' : 344.04,
        '207' : 336.80,
        '208' : 330.25,
        '209' : 324.46,
        '210' : 319.39,
        '211' : 315,
        '212' : 310.60,
        '212' : 305.53,
        '214' : 299.74,
        '215' : 293.19,
        '216' : 285.94,
        '217' : 278.12,
        '218' : 270,
        '219' : 261.86,
        '220' : 254.05,
        '221' : 246.80,
        '222' : 240.25,
        '223' : 234.46,
        '224' : 229.39,
        '225' : 225,
        '226' : 221.18,
        '227' : 216.86,
        '228' : 212,
        '229' : 206.56,
        '230' : 200.55,
        '231' : 194.03,
        '232' : 187.12,
        '233' : 180,
        '234' : 172.87,
        '235' : 165.96,
        '236' : 159.44,
        '237' : 153.43,
        '238' : 147.99,
        '239' : 143.12,
        '240' : 128.81,
        '241' : 125,
        '242' : 121.18,
        '243' : 126.86,
        '244' : 122,
        '245' : 116.56,
        '246' : 110.55,
        '247' : 104.03,
        '248' : 97.12,
        '249' : 90,
        '250' : 82.87,
        '251' : 75.96,
        '252' : 69.44,
        '253' : 63.43,
        '254' : 57.99,
        '255' : 53.12,
        '256' : 48.81,
        '257' : 45,
        '258' : 41.18,
        '259' : 36.86,
        '260' : 32,
        '261' : 26.56,
        '262' : 20.55,
        '263' : 14.03,
        '264' : 7.12,
        '265' : 0,
        '266' : 352.87, 
        '267' : 345.96,
        '268' : 339.44,
        '269' : 333.43,
        '270' : 327.99,
        '271' : 323.12,
        '272' : 318.81,
        '273' : 315,
        '274' : 311.18,
        '275' : 306.86,
        '276' : 302,
        '277' : 296.56,
        '278' : 290.55,
        '279' : 284.03,
        '280' : 277.12,
        '281' : 270,
        '282' : 262.87,
        '283' : 255.96,
        '284' : 249.44,
        '285' : 243.43,
        '286' : 237.99,
        '287' : 233.12,
        '288' : 228.81,
        '289' : 225,
        '290' : 221.63,
        '291' : 217.87,
        '292' : 212.69,
        '293' : 209.05,
        '294' : 203.96,
        '295' : 198.43,
        '296' : 192.52,
        '297' : 186.34,
        '298' : 180,
        '299' : 173.65,
        '300' : 167.47,
        '301' : 161.56,
        '302' : 156.03,
        '303' : 150.94,
        '304' : 146.31,
        '305' : 142.12,
        '306' : 128.36,
        '307' : 125,
        '308' : 121.63,
        '309' : 127.87,
        '310' : 123.69,
        '311' : 119.05,
        '312' : 112.96,
        '312' : 108.43,
        '314' : 102.52,
        '315' : 96.34,
        '316' : 90,
        '317' : 83.65,
        '318' : 77.47,
        '319' : 71.56,
        '320' : 66.03,
        '321' : 60.94,
        '322' : 56.30,
        '323' : 52.12,
        '324' : 48.36,
        '325' : 45,
        '326' : 41.63,
        '327' : 37.87,
        '328' : 33.69,
        '329' : 29.05,
        '330' : 23.96,
        '331' : 18.43,
        '332' : 12.52,
        '333' : 6.34,
        '334' : 0 ,
        '335' : 353.65,
        '336' : 347.47,
        '337' : 341.56,
        '338' : 336.03,
        '339' : 330.94,
        '340' : 326.30,
        '341' : 322.12,
        '342' : 318.36,
        '343' : 315,
        '344' : 311.63,
        '345' : 307.87,
        '346' : 303.69,
        '347' : 299.05,
        '348' : 293.96,
        '349' : 288.43,
        '350' : 282.52,
        '351' : 276.34,
        '352' : 270,
        '353' : 263.65,
        '354' : 257.47,
        '355' : 251.56,
        '356' : 246.03,
        '357' : 240.94,
        '358' : 236.30,
        '359' : 232.12,
        '360' : 228.36,
        '361' : 225,


    }


    A = CELLS[A]
    B = CELLS[B]
    C = CELLS[C]
    D = CELLS[D]
    E = CELLS[E]
    F = CELLS[F]
    G = CELLS[G]
    H = CELLS[H]
    I = CELLS[I]
    J = CELLS[J]

    A_B_ROTATION = A - B
    B_C_ROTATION = B - C
    C_D_ROTATION = C - D
    D_E_ROTATION = D - E
    E_F_ROTATION = E - F 
    F_G_ROTATION = F - G
    G_H_ROTATION = G - H
    H_I_ROTATION = H - I
    I_J_ROTATION = I - J


    print(A)
    print(B)
    print(C)
    print(D)
    print(E)
    print(F)
    print(G)
    print(H)
    print(I)
    print(J)


    plt.plot(price.values)
    plt.scatter(current_idx, current_pat, c='r')
    plt.show()


