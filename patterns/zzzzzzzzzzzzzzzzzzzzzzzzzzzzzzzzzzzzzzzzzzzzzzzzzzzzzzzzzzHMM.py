
#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr


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
#from sklearn.externals import joblib
import ta


#import xgboost
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
from sklearn import  metrics, model_selection
#from xgboost.sklearn import XGBClassifier



ticker = [
# S AND P 500       14450

                                       # 'ABMD' ,

'EURUSD=X',

]


def get_best_hmm_model(X, max_states, max_iter = 10000):
    best_score = -(10 ** 10)
    best_state = 0
    
    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components = state, random_state = 100,
                                covariance_type = "diag", n_iter = max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state
    
    best_model = GaussianHMM(n_components = best_state, random_state = 100,
                                covariance_type = "diag", n_iter = max_iter).fit(X)
    return best_model

# Normalized st. deviation
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes and price
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)

# General plots of hidden states
def plot_hidden_states(hmm_model, data, X, column_price):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(hmm_model.n_components, 3, figsize = (15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    hidden_states = model.predict(X)
    
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c = 'grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c = colour)
        ax[0].set_title("{0}th hidden state".format(i))
        ax[0].grid(True)
        
        ax[1].hist(data["future_return"][mask], bins = 30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title("future return distrbution at {0}th hidden state".format(i))
        ax[1].grid(True)
        
        ax[2].plot(data["future_return"][mask].cumsum(), c = colour)
        ax[2].set_title("cummulative future return at {0}th hidden state".format(i))
        ax[2].grid(True)
        
    plt.tight_layout()


def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m, m + h

def compare_hidden_states(hmm_model, cols_features, conf_interval, iters = 1000):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize = (15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    
    for i in range(0, model.n_components):
        mc_df = pd.DataFrame()
    
        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features
    
        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color = colours[i])
            axs[k][i].set_title(cols_features[k] + " (state " + str(i) + "): " + str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)
            
    plt.tight_layout()








for x in ticker:
    print(x)
    #data = pdr.get_data_yahoo(x,  period = "30d",  interval = "90m")
    data =  pdr.get_data_yahoo(x, interval = "1d", start="1990-01-01", end="2021-10-16")

    data = data.reset_index()

    #data = data[:-22]

    print(data.tail(30))

    datasetINDICATORS = pd.DataFrame(columns= ['Open', 'High', 'Low', 'Close', 'Volume'])

    #datasetINDICATORS['Timestamp'] =  data['Datetime']
    datasetINDICATORS['Open'] = data['Open']
    datasetINDICATORS['High'] = data['High']
    datasetINDICATORS['Low'] = data['Low']
    datasetINDICATORS['Close'] = data['Close']
    datasetINDICATORS['Volume'] = data['Volume']

    # Add all ta features
    datasetINDICATORS = ta.add_all_ta_features(
        datasetINDICATORS, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )

    
    dataset = pd.DataFrame(columns= ['Date', 'High', 'Low', 'Mid', 'Last', 'Volume']) #'Bid', 'Ask',  

    dataset['Date'] =  data['Date']
    dataset['High'] = data['High']
    dataset['Low'] = data['Low']
    dataset['Mid'] = data['Open']
    dataset['Last'] = data['Close']
    dataset['Volume'] = data['Volume']
    

    # Feature params
    future_period = 6
    std_period = 10
    ma_period = 10
    price_deviation_period = 10
    volume_deviation_period = 10
    column_price = 'Last'
    column_high = 'High'
    column_low = 'Low'
    column_volume = 'Volume'


    # Create features
    ### CHANGES
    cols_features = [

        'last_return', 
        #'std_normalized',
        #'ma_ratio', 
        #'price_deviation',
        #'volume_deviation',

        #'volume_obv',
        #'volume_cmf',
        #'volume_fi',   MAYBE
        #'volume_em',
        #'volume_sma_em',
        #'volume_vpt',
        #'volume_nvi',
        'volatility_atr',
        #'volatility_bbm',
        #'volatility_bbh',
        #'volatility_bbl',
        #'volatility_bbw',
        #'volatility_kcc',
        #'volatility_kch',
        #'volatility_kcl',
        #'volatility_dcl',
        #'volatility_dch',
            #'volatility_dchi',
            #'volatility_dcli',
        'trend_macd',
        #'trend_macd_signal',
        #'trend_macd_diff',
        #'trend_ema_fast',
        #'trend_ema_slow',
        #'trend_adx',
        #'trend_adx_pos',
        #'trend_adx_neg',
        #'trend_vortex_ind_pos', 
        #'trend_vortex_ind_neg',
        #'trend_vortex_ind_diff',
        #'trend_trix',
        #'trend_mass_index',
        #'trend_cci', MAYBE
        #'trend_dpo', 
        #'trend_kst', 
        #'trend_kst_sig',
        #'trend_kst_diff', 
        #'trend_ichimoku_a',  
        #'trend_ichimoku_b',
        #'trend_visual_ichimoku_a',  
        #'trend_visual_ichimoku_b',
        #'trend_aroon_up',
        #'trend_aroon_down',  
        #'trend_aroon_ind',

            #'trend_psar',  
            #'trend_psar_up',
            #'trend_psar_down',  
            #'trend_psar_up_indicator',  
            #'trend_psar_down_indicator',
        #'momentum_rsi',
        #'momentum_mfi', MAYBE
        #'momentum_tsi',  
        #'momentum_uo', 
        'momentum_stoch',
        #'momentum_stoch_signal',  
        #'momentum_wr',
        #'momentum_ao',
        #'momentum_kama',
        #'momentum_roc',     
        #'others_dr',  
        #'others_dlr',   
        #'others_cr',  


        ]



    dataset['last_return'] = dataset[column_price].pct_change()
    dataset['std_normalized'] = dataset[column_price].rolling(std_period).apply(std_normalized)
    #dataset['ma_ratio'] = dataset[column_price].rolling(std_period).apply(ma_ratio)  
    dataset['ma_ratio'] = dataset['Last']
    dataset['price_deviation'] = dataset['High'] - dataset['Low']
    dataset['volume_deviation'] = dataset['Volume']

    dataset["pivot_point"] = (dataset['High'] + dataset['Low'] + dataset['Last']) / 3
    dataset["pivH1"] = (2 * dataset["pivot_point"]) - dataset['Low']
    dataset["pivL1"] = (2 * dataset["pivot_point"]) + dataset['High']
    dataset["pivH2"] = dataset["pivot_point"] + (dataset['High'] - dataset['Low'])
    dataset["pivL2"] = dataset["pivot_point"] - (dataset['High'] - dataset['Low'])
    dataset["pivH3"] = dataset["High"] + 2 * (dataset["pivot_point"] - dataset['Low'])
    dataset["pivL3"] = dataset["Low"] - 2 * (dataset["High"] - dataset["pivot_point"])


    dataset['volume_obv'] = datasetINDICATORS['volume_obv']
    dataset['volume_cmf'] = datasetINDICATORS['volume_cmf']
    dataset['volume_fi'] = datasetINDICATORS['volume_fi']
    dataset['volume_em'] = datasetINDICATORS['volume_em']
    dataset['volume_sma_em'] = datasetINDICATORS['volume_sma_em']
    dataset['volume_vpt'] =  datasetINDICATORS['volume_vpt']
    dataset['volume_nvi'] =  datasetINDICATORS['volume_nvi']
    dataset['volatility_atr'] = datasetINDICATORS['volatility_atr']
    dataset['volatility_bbm'] = datasetINDICATORS['volatility_bbm']
    dataset['volatility_bbh'] = datasetINDICATORS['volatility_bbh']
    dataset['volatility_bbl'] = datasetINDICATORS['volatility_bbl']
    dataset['volatility_bbw'] = datasetINDICATORS['volatility_bbw']
    dataset['volatility_kcc'] = datasetINDICATORS['volatility_kcc']
    dataset['volatility_kch'] = datasetINDICATORS['volatility_kch']
    dataset['volatility_kcl'] = datasetINDICATORS['volatility_kcl']
    dataset['volatility_dcl'] = datasetINDICATORS['volatility_dcl']
    dataset['volatility_dch'] = datasetINDICATORS['volatility_dch']


    dataset['trend_macd'] =  datasetINDICATORS['trend_macd']
    dataset['trend_macd_signal'] =  datasetINDICATORS['trend_macd_signal']
    dataset['trend_macd_diff'] = datasetINDICATORS['trend_macd_diff']
    dataset['trend_ema_fast'] = datasetINDICATORS['trend_ema_fast']
    dataset['trend_ema_slow'] =  datasetINDICATORS['trend_ema_slow']
    dataset['trend_adx'] = datasetINDICATORS['trend_adx']
    dataset['trend_adx_pos'] = datasetINDICATORS['trend_adx_pos']
    dataset['trend_adx_neg'] = datasetINDICATORS['trend_adx_neg']
    dataset['trend_vortex_ind_pos'] = datasetINDICATORS['trend_vortex_ind_pos'] 
    dataset['trend_vortex_ind_neg'] = datasetINDICATORS['trend_vortex_ind_neg']
    dataset['trend_vortex_ind_diff'] = datasetINDICATORS['trend_vortex_ind_diff']
    dataset['trend_trix'] = datasetINDICATORS['trend_trix']
    dataset['trend_mass_index'] = datasetINDICATORS['trend_mass_index']
    dataset['trend_cci'] = datasetINDICATORS['trend_cci']
    dataset['trend_dpo'] =  datasetINDICATORS['trend_dpo'] 
    dataset['trend_kst'] = datasetINDICATORS['trend_kst'] 
    dataset['trend_kst_sig'] = datasetINDICATORS['trend_kst_sig']
    dataset['trend_kst_diff']  =  datasetINDICATORS['trend_kst_diff']
    dataset['trend_ichimoku_a'] =  datasetINDICATORS['trend_ichimoku_a']  
    dataset['trend_ichimoku_b'] = datasetINDICATORS['trend_ichimoku_b']
    dataset['trend_visual_ichimoku_a']  = datasetINDICATORS['trend_visual_ichimoku_a'] 
    dataset['trend_visual_ichimoku_b'] = datasetINDICATORS['trend_visual_ichimoku_b'] 
    dataset['trend_aroon_up'] =  datasetINDICATORS['trend_aroon_up']
    dataset['trend_aroon_down'] =  datasetINDICATORS['trend_aroon_down']   
    dataset['trend_aroon_ind'] = datasetINDICATORS['trend_aroon_ind'] 
    dataset['momentum_rsi'] = datasetINDICATORS['momentum_rsi']

    dataset['momentum_tsi'] = datasetINDICATORS['momentum_tsi'] 
    dataset['momentum_uo']  = datasetINDICATORS['momentum_uo']
    dataset['momentum_stoch'] =  datasetINDICATORS['momentum_stoch']
    dataset['momentum_stoch_signal'] = datasetINDICATORS['momentum_stoch_signal'] 
    dataset['momentum_wr'] = datasetINDICATORS['momentum_wr']
    dataset['momentum_ao'] = datasetINDICATORS['momentum_ao']
    dataset['momentum_kama'] = datasetINDICATORS['momentum_kama']
    dataset['momentum_roc'] = datasetINDICATORS['momentum_roc'] 
    dataset['others_dr'] = datasetINDICATORS['others_dr'] 
    dataset['others_dlr'] = datasetINDICATORS['others_dlr']    
    dataset['others_cr'] = datasetINDICATORS['others_cr'] 


    dataset["future_return"] = dataset[column_price].pct_change(future_period).shift(-future_period)
    

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    train_set = dataset[cols_features]

    train_set = train_set.dropna()

    # Add the features we want to use

    dataset['close/pivH1'] = dataset['Last'] / dataset['pivH1']
    dataset['close/pivL1'] = dataset['Last'] / dataset['pivL1']

    dataset['close/pivH2'] = dataset['Last'] / dataset['pivH2']

    dataset['close/pivL2'] = dataset['Last'] / dataset['pivL2']


    dataset['close/pivH3'] = dataset['Last'] / dataset['pivH3']
    dataset['close/pivL3'] = dataset['Last'] / dataset['pivL3']

    dataset['pivL1/pivH1'] = dataset['pivL1'] / dataset['pivH1']
    dataset['pivL2/pivH2'] = dataset['pivL2'] / dataset['pivH2']

    dataset['high/pivH1'] = dataset['High'] / dataset['pivH1']
    dataset['low/pivH1'] = dataset['Low'] / dataset['pivH1']
    dataset['high/pivL1'] = dataset['High'] / dataset['pivL1']


    dataset['low/pivL1'] = dataset['Low'] / dataset['pivL1']

    dataset['close/prevClose'] = dataset['Last'] / dataset['Last'].shift(1)

    # Below are the things we are interested in predicting:

    dataset['next_candle_size'] = abs(dataset['Last'].shift(-1) - dataset['Last']) / dataset['Last']

    # Result is -1, 1, or 0 at the mo - its not binary!! So we just want to know if it is 1 or 0

    dataset['next_candle_color'] = np.where(dataset['Last'].shift(-1) > dataset['Last'], 1, -1)

    #data = dataset[['Date', 'Mid', 'High', 'Low', 'Last', 'close/prevClose','low/pivL1', 'close/pivH3',
    #            'close/pivH1', 'close/pivL2', 'close/pivL1', 'close/pivH2', 'high/pivL1', 'close/pivL3', 
    #            'low/pivH1', 'pivL2/pivH2', 'high/pivH1', 'next_candle_color', 'next_candle_size'
    #            ]]

    data = dataset[['Date', 'Mid', 'High', 'Low', 'Last', 'close/prevClose','low/pivL1', 'close/pivH3',
                'close/pivH1', 'close/pivL2', 'close/pivL1', 'close/pivH2', 'high/pivL1', 'close/pivL3', 
                'low/pivH1', 'pivL2/pivH2', 'high/pivH1', 'next_candle_color', 'next_candle_size'
                ]]

    data = data.dropna()


    # XG Classify
    trainning_data = data

    df_xg = trainning_data

    df_xg.dropna(axis=0, inplace=True)

    X = df_xg.iloc[:,5:17]
    y = df_xg.iloc[:,-2]

    params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1,
    'silent': 1,
    'n_estimators': 5
    }
    




    #print(x)
    #print(prediction_xg)
    ####################################

    print(train_set)

    try:
        model = get_best_hmm_model(X = train_set, max_states = 3, max_iter = 1000000)
    except:
        print('means weight')

    print("Best model with {0} states ".format(str(model.n_components)))

    #    plot_hidden_states(model, dataset.reset_index(), train_set, column_price)
    #    compare_hidden_states(hmm_model=model, cols_features=cols_features, conf_interval=0.95)

    

    predictionDataset = dataset.iloc[-1:]

    prediction = model.predict(predictionDataset[cols_features])

    # LOGIC SHORT WHEN 0; NO POSITION WHEN 1; LONG WHEN 2;
    print(x)
    print(train_set.size)
    #print(len(prediction))
    print(sum(prediction))

