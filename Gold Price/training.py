from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)


def drop_per(df):
    df["Change %"] = df["Change %"].str.replace(pat=r'[%]', repl=r'', regex=True)
    df["Change %"] = pd.to_numeric(df["Change %"])
    return df

def drop_stick(df):
    df["Vol."] = df["Vol."].str.replace(pat=r'-', repl=r'', regex=True)
    return df

def drop_K(df):
    df["Vol."] = df["Vol."].str.replace(pat=r'K', repl=r'', regex=True)
    df["Vol."] = pd.to_numeric(df["Vol."])
    return df

def missingvalue(raw, type="previous_data"):
    raw = drop_per(raw)
    raw = drop_stick(raw)
    raw = drop_K(raw)
    data = raw  # .drop("Date", axis=1)

    data = data[::-1]
    data = data.reset_index(drop=True)

    if type == "previous_data":  # fill with previous data
        data = data.fillna(axis=0, method='ffill')

    elif type == "next_data":
        data == data.fillna(axis=0, method='bfill')

    return data

def windowing(data, i):
    windows = [data[k:k + i] for k in range(len(data) - i + 1)]
    return windows

def train(x_path, p_path, p, i, n_components):

    ''' Get Training Datasets '''
    x_data = pd.read_csv(x_path, thousands=',')
    p_data = pd.read_csv(p_path, thousands=',')

    nullnum = x_data['MACD'].isnull().sum()
    x_data = x_data.dropna(axis=0)
    p_data = p_data.iloc[:len(p_data) - nullnum]

    x_data = x_data[::-1]
    p_data = p_data[::-1]

    m = len(x_data)
    print(m)

    itr_num = int((m - p - 3 * i - 19) / (i + 9) + 1)
    init_idx = m - 3 * i - p - 19 - (i + 9) * (itr_num - 1)
    print(itr_num, init_idx)

    x_data = x_data.iloc[:len(x_data) - i - 10]
    p_data = p_data.iloc[:len(p_data) - i - 10]

    x_data = x_data.reset_index(drop=True)
    p_data = p_data.reset_index(drop=True)

    ''' Start New Period '''
    ''' Get Data '''
    train = x_data[-i - p + 1:].drop("Date", axis=1).iloc[:, [0, 1, 3, 6, 8, 10, 13, 14, 15]]  # .values.tolist()

    ''' Windowing Training Data '''
    train_windows = []
    for time in range(len(train) - i + 1):
        feat = []
        for k in range(len(train.iloc[0])):
            f = train.iloc[:, k]
            f = f[time:time + i]
            feat = np.concatenate((feat, f[::-1]))
        train_windows.append(feat)

    ''' Initialize and Train Model '''
    model = GaussianHMM(n_components)
    model.fit(train_windows)  # train model with train data

    filename = 'team02_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    x_path = "D:/2021/학교/1학기/데이터기반학습/LFD_Project2/data/commodities/data.csv"
    p_path = "D:/2021/학교/1학기/데이터기반학습/LFD_Project2/data/commodities/Gold.csv"
    train(x_path, p_path, 1800, 5, 4)  # x_path, p_path, p, i, n_components
