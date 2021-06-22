import os
import pandas as pd
import numpy as np


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = 'D:/2021/학교/1학기/데이터기반학습/LFD_Project2/data/commodities'
    currency_dir = 'D:/2021/학교/1학기/데이터기반학습/LFD_Project2/data/currencies'

    if symbol in ['AUD_KRW', 'CNY_KRW', 'EUR_KRW', 'GBP_KRW', 'HKD_KRW', 'JPY_KRW', 'USD_KRW']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path

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

    # data = data[::-1] DO NOT BANJEON
    data = data.reset_index()

    if type == "previous_data":  # fill with previous data
        data = data.fillna(axis=0, method='ffill')

    elif type == "next_data":
        data == data.fillna(axis=0, method='bfill')
    #
    return data

def create_diff(df, type="Price"):
    # df["Diff"] = df[type].diff()
    df.insert(1, "Diff", df[type].diff(periods=-1))
    return df

def create_diff_y(df, type="Price_y"):
    df.insert(1, "Diff_y", df[type].diff(periods=-1))
    return df

def drop_per(df):
    df["Change %"] = df["Change %"].str.replace(pat=r'[%]', repl=r'', regex=True)
    df["Change %"] = pd.to_numeric(df["Change %"])
    return df

def ROC5(data, step, param="Price"):
    exData = data  # .head(10)
    exData["5RoC"] = np.nan

    for index in range(len(exData) - step):
        Price_t = exData.loc[index, param]
        Price_t_step = exData.loc[index + step, param]
        RoC = (float(Price_t) - float(Price_t_step)) / float(Price_t_step)
        exData["5RoC"][index] = RoC
    return exData

def ROC10(data, step, param="Price"):
    exData = data  # .head(10)
    exData["10RoC"] = np.nan

    for index in range(len(exData) - step):
        Price_t = exData.loc[index, param]
        Price_t_step = exData.loc[index + step, param]
        RoC = (float(Price_t) - float(Price_t_step)) / float(Price_t_step)
        exData["10RoC"][index] = RoC
    return exData

def MA5(data, step, param="Price"):
    exData = data  # .head(10)
    exData["5MA"] = np.nan

    for index in range(len(exData) - step):
        list = []
        for i in range(step):
            list.append(float(exData['Price'][index + i]))
        exData["5MA"][index] = float(sum(list) / step)

    return exData

def MA10(data, step, param="Price"):
    exData = data  # .head(10)
    exData["10MA"] = np.nan

    for index in range(len(exData) - step):
        list = []
        for i in range(step):
            list.append(float(exData['Price'][index + i]))
        exData["10MA"][index] = float(sum(list) / step)

    return exData

def Bias5(data, step, param="Price"):
    # NOTE: This function must be called after function "MovingAverage"
    exData = data  # .head(10)
    exData["5Bias"] = np.nan

    for index in range(len(data) - step):
        exData["5Bias"][index] = (float(exData[param][index]) - exData["5MA"][index]) / exData["5MA"][index] * 100

    return exData

def Bias10(data, step, param="Price"):
    # NOTE: This function must be called after function "MovingAverage"
    exData = data  # .head(10)
    exData["10Bias"] = np.nan

    for index in range(len(data) - step):
        exData["10Bias"][index] = (float(exData[param][index]) - exData["10MA"][index]) / exData["10MA"][index] * 100

    return exData

def RSV(data, step, param="Price"):  # Used parameters: Price, High, Low
    exData = data  # .head(10)
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
    exData = data  # .head(10)
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
    exData = data  # .head(10)
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

def make_indicators():
    gold_data = pd.read_csv(get_data_path("Gold"), thousands=',', index_col='Date', parse_dates=True, na_values=['nan'])
    # silver_data = pd.read_csv(get_data_path("Silver"), thousands=',', index_col='Date',parse_dates=True, na_values=['nan'])

    gold_data = missingvalue(gold_data)
    # silver_data = missingvalue(silver_data)

    gold_data = create_diff(gold_data)

    gold_data_ROC5 = ROC5(gold_data, 5)
    gold_data_ROC10 = ROC10(gold_data_ROC5, 10)
    gold_data_MA5 = MA5(gold_data_ROC10, 5)
    gold_data_MA10 = MA10(gold_data_MA5, 10)
    gold_data_Bias5 = Bias5(gold_data_MA10, 5)
    gold_data_Bias10 = Bias10(gold_data_Bias5, 10)
    gold_data_RSV = RSV(gold_data_Bias10, 9)
    gold_data_Stochastic = Stochastic(gold_data_RSV, 9)
    gold_data_MACD = MACD(gold_data_Stochastic)
    gold_data_WR = WR(gold_data_MACD, 10)
    gold_data_RSI = RSI(gold_data_WR, 14)

    gold_data = gold_data_RSI[['Diff', 'Price', 'Change %', '5MA', '5Bias', 'K', 'MACD', 'WR', 'RSI']]

    return gold_data

def make_features():
    x_data = make_indicators()
    p_data = make_indicators()["Price"]
    nullnum = x_data['MACD'].isnull().sum()
    x_data = x_data.dropna(axis=0)
    p_data = p_data.iloc[:len(p_data) - nullnum]

    x_data = x_data[::-1]
    p_data = p_data[::-1]

    x_data = x_data.reset_index(drop=True)
    p_data = p_data.reset_index(drop=True)

    i = 5

    target_price = p_data[-10:]
    target_price = target_price.reset_index(drop=True)
    past_price = p_data[-11:-1]
    past_price = past_price.reset_index(drop=True)

    test_x = []
    for time in range(10):
        daily_feature = x_data[-10 - i + time:-10 + time][::-1]
        gold_diff = daily_feature["Diff"]
        gold_price = daily_feature["Price"]
        gold_change = daily_feature['Change %']
        gold_ma = daily_feature['5MA']
        gold_bias = daily_feature['5Bias']
        gold_k = daily_feature['K']
        gold_macd = daily_feature['MACD']
        gold_wr = daily_feature['WR']
        gold_rsi = daily_feature['RSI']
        test_feature = np.concatenate((gold_diff, gold_price, gold_change, gold_ma,
                                       gold_bias, gold_k, gold_macd, gold_wr, gold_rsi))
        test_x.append(test_feature)

    return (test_x, past_price, target_price)