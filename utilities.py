# source: Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import datetime

TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"
now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/202112"
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
TESTING_DATA_FILE = "test.csv"


def preprocess_data(file_name):

    df = pd.read_csv(file_name)
    df = df[df.datadate>=20090000]
    df_preprocess = calcualte_price(df)
    df_final=add_technical_indicator(df_preprocess)
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def data_split(df,start,end):

    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'])
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):

    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'])
    return data

def add_technical_indicator(df):
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df

