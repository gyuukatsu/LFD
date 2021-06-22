from sklearn import neural_network as NN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
import pickle
import numpy as np
import pandas as pd
import os
import glob

x_path = "D:/2021/학교/1학기/데이터기반학습/LFD_Project1/data/currencies/data.csv"
y_path = "D:/2021/학교/1학기/데이터기반학습/LFD_Project1/data/currencies/USD_KRW.csv"


def windowing_x(data, i):
    windows = [data[k:k + i] for k in range(len(data) - i + 1)]  # Note the range
    return windows


def windowing_y(data, i):
    windows = [data[k:k + 10] for k in range(len(data) - 10 + 1)]  # Note the range
    return windows


def train(x_path, y_path, p, i, param):
    ''' Get Training Datasets '''
    # Import Training csv file and convert them into DataFrame
    # Index of the row from 0 to len(DataFrame)
    # First column of the DataFrame is dates
    x_data = pd.read_csv(x_path, thousands=',')  # Import Merged Data
    y_data = pd.read_csv(y_path, thousands=',')

    nullnum = x_data['MACD'].isnull().sum()
    x_data = x_data.dropna(axis=0)
    y_data = y_data.iloc[:len(y_data) - nullnum]

    x_data = x_data[::-1]
    y_data = y_data[::-1]

    x_data = x_data.iloc[:len(x_data) - i - 10]
    y_data = y_data.iloc[:len(y_data) - i - 10]

    x_data = x_data.reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)

    ''' Set Model Parameters '''
    itr_num = int((len(x_data) - p - i - 19) / 10) + 1
    init_idx = len(x_data) - p - i - 9 - 10 * itr_num

    print("Period: ", p, "Input Days: ", i)
    print("Hidden Layer Sizes:", param)

    n = itr_num
    ''' Begin Training '''
    model = NN.MLPRegressor(hidden_layer_sizes=param, alpha=0.0001, max_iter=10000, verbose=False)
    train_x = x_data[10 * (n - 1) + 10 + init_idx:10 * (n - 1) + p + i + 8 + 1 + init_idx].drop("Date", axis=1).iloc[:,
              [0, 8, 9, 10, 12, 13, 14, 19, 20, 21, 22]].values.tolist()  # plus ones for index convention
    train_y = y_data[10 * (n - 1) + i + 10 + init_idx:10 * (n - 1) + p + i + 18 + 1 + init_idx].drop("Date",
                                                                                                     axis=1).iloc[:,
              0].values.tolist()  # iloc index can be changed

    ''' Get Windows '''
    window_x = windowing_x(train_x, i)
    window_y = windowing_y(train_y, i)
    # Trim some data for sklearn function
    nsamples, nx, ny = np.array(window_x).shape
    window_x = np.array(window_x).reshape((nsamples, nx * ny))
    ''' Train Regressor '''
    model.fit(window_x, window_y)

    filename = 'team02_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    train(x_path, y_path, 2100, 10, (20, 20, 20))