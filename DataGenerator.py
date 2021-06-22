import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = 'D:/2021/학교/1학기/데이터기반학습/LFD_Project1/data/commodities'
    currency_dir = 'D:/2021/학교/1학기/데이터기반학습/LFD_Project1/data/currencies'

    if symbol in ['AUD_KRW', 'CNY_KRW', 'EUR_KRW', 'GBP_KRW', 'HKD_KRW', 'JPY_KRW', 'USD_KRW']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def MA5(data, step, param="Price"):
    exData = data
    exData["5MA"] = np.nan

    for index in range(len(exData) - step):
        list = []
        for i in range(step):
            list.append(float(exData['Price'][index + i]))
        exData["5MA"][index] = float(sum(list) / step)

    return exData


def MA10(data, step, param="Price"):
    exData = data
    exData["10MA"] = np.nan

    for index in range(len(exData) - step):
        list = []
        for i in range(step):
            list.append(float(exData['Price'][index + i]))
        exData["10MA"][index] = float(sum(list) / step)

    return exData


def Bias5(data, step, param="Price"):
    # NOTE: This function must be called after function "MovingAverage"
    exData = data
    exData["5Bias"] = np.nan

    for index in range(len(data) - step):
        exData["5Bias"][index] = (float(exData[param][index]) - exData["5MA"][index]) / exData["5MA"][index] * 100

    return exData


def Bias10(data, step, param="Price"):
    # NOTE: This function must be called after function "MovingAverage"
    exData = data
    exData["10Bias"] = np.nan

    for index in range(len(data) - step):
        exData["10Bias"][index] = (float(exData[param][index]) - exData["10MA"][index]) / exData["10MA"][index] * 100

    return exData


def RSV(data, step, param="Price"):  # Used parameters: Price, High, Low
    exData = data
    exData["RSV"] = np.nan
    for i in range(len(exData) - step):
        list_high = []
        list_low = []
        for j in range(step):
            list_high.append(float(exData["High"][i + j]))
            list_low.append(float(exData["Low"][i + j]))
        if ((max(list_high) - min(list_low)) == 0):
            print(i)
        exData["RSV"][i] = (float(exData[param][i]) - min(list_low)) / (max(list_high) - min(list_low)) * 100

    return exData


def Stochastic(data, step):
    # NOTE: This function must be called after function "RSV"
    exData = data
    exData["K"], exData["D"], exData["J"] = 0.0, 0.0, 0.0
    for i in range(len(exData) - step):
        exData["K"][len(exData) - step - 1 - i] = exData["K"][len(exData) - step - i] * 2 / 3 + 1 / 3 * exData["RSV"][
            len(exData) - step - 1 - i]
        exData["D"][len(exData) - step - 1 - i] = exData["D"][len(exData) - step - i] * 2 / 3 + 1 / 3 * exData["K"][
            len(exData) - step - 1 - i]
        exData["J"][len(exData) - step - 1 - i] = exData["D"][len(exData) - step - 1 - i] * 3 - 2 * exData["K"][
            len(exData) - step - 1 - i]
    return exData


def MACD(data):
    exData = data
    exData["DI"], exData["EMA12"], exData["EMA26"], exData["DIF"], exData[
        "MACD"] = np.nan, np.nan, np.nan, np.nan, np.nan

    for i in range(len(exData)):
        exData["DI"][i] = (float(exData["High"][i]) + float(exData["Low"][i]) + 2 * float(exData["Price"][i])) / 4

    for i in range(len(exData) - 11):
        list = []
        for j in range(12):
            list.append(float(exData["DI"][i + j]))
            exData["EMA12"][i] = sum(list) / 12

    for i in range(len(exData) - 25):
        list = []
        for j in range(26):
            list.append(float(exData["DI"][i + j]))
            exData["EMA26"][i] = sum(list) / 26

    for i in range(len(exData) - 25):
        exData["DIF"][i] = float(exData["EMA12"][i]) - float(exData["EMA26"][i])

    for i in range(len(exData) - 33):
        list = []
        for j in range(9):
            list.append(exData["DIF"][i + j])
        exData["MACD"][i] = sum(list) / 9
    return exData


def WR(data, step):  # Used parameters: Price, High, Low
    exData = data
    exData["WR"] = np.nan
    for i in range(len(exData) - step):
        list_high = []
        list_low = []
        for j in range(step):
            list_high.append(float(exData["High"][i + j]))
            list_low.append(float(exData["Low"][i + j]))
        exData["WR"][i] = (max(list_low) - float(exData["Price"][i])) / (max(list_high) - min(list_low))
    return exData


def RSI(data, step):
    exData = data
    exData["RSI"] = np.nan
    for i in range(len(exData) - step):
        list_up = []
        list_down = []
        for j in range(step):
            if float(exData["Price"][i + j]) > float(exData["Price"][i + j + 1]):
                list_up.append(float(exData["Price"][i + j]) - float(exData["Price"][i + j + 1]))
            elif float(exData["Price"][i + j]) < float(exData["Price"][i + j + 1]):
                list_down.append(float(exData["Price"][i + j + 1]) - float(exData["Price"][i + j]))
            else:
                pass
        if len(list_up) != 0 and len(list_down) != 0:
            exData["RSI"][i] = (sum(list_up) / len(list_up)) / (
                        sum(list_up) / len(list_up) + sum(list_down) / len(list_down)) * 100
        elif len(list_up) == 0 and len(list_down) != 0:
            exData["RSI"][i] = 0.0
        elif len(list_up) != 0 and len(list_down) == 0:
            exData["RSI"][i] = 100.0
        else:
            exData["RSI"][i] = 50.0
    return exData


def DIFF(data, param="Price"):
    exData = data
    exData["diff"] = np.nan
    for i in range(len(exData) - 1):
        diff = - exData.loc[i + 1, param] + exData.loc[i, param]
        exData["diff"][i] = diff
    return exData


def make_indicators(data):
    data_MA5 = MA5(data, 5)
    data_MA10 = MA10(data_MA5, 10)
    data_Bias5 = Bias5(data_MA10, 5)
    data_Bias10 = Bias10(data_Bias5, 10)
    data_RSV = RSV(data_Bias10, 9)
    data_Stochastic = Stochastic(data_RSV, 9)
    data_MACD = MACD(data_Stochastic)
    data_WR = WR(data_MACD, 10)
    data_RSI = RSI(data_WR, 14)
    data_diff = DIFF(data_RSI)

    fi_data = data_diff.drop(['Open', 'High', 'Low', 'Change %', '5MA', 'RSV', 'DI', 'EMA12', 'EMA26', 'DIF'], axis=1)

    return fi_data


def make_features(data):
    data_indicators = make_indicators(data)

    nullnum = data_indicators['MACD'].isnull().sum()
    x_data = data_indicators.dropna(axis=0)
    y_data = data.iloc[:len(data) - nullnum]

    x_data = x_data[::-1]
    y_data = y_data[::-1]

    x_data = x_data.reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)

    ''' Split Test Datasets '''
    test_x = x_data.iloc[-20:-10,:].drop("Date", axis=1).values.tolist()
    test_y = y_data.tail(10).drop("Date", axis=1).iloc[:, 0].values.tolist()

    nsamples, nx, ny = np.array([test_x]).shape
    test_x = np.array(test_x).reshape((nsamples, nx * ny))

    return (test_x, test_y)