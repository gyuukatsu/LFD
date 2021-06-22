import pandas as pd
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings(action='ignore')


def multiply_two(val_str):
    return int(val_str) * 2


def divide_two(val_str):
    return int(int(val_str) / 2)


def multiply_1_5862(val_str):
    return int(int(val_str) * 1.5862)


def divide_1_5862(val_str):
    return int(int(val_str) / 1.5862)


def multiply_0_2(val_str):
    return int(int(val_str) * 0.2)


def divide_0_2(val_str):
    return int(int(val_str) * 5)


def OpenScale(data):
    data["OpenScale"] = np.nan
    for idx in range(0, len(data)):
        data.loc[data.index[idx], "OpenScale"] = float(data["Open"][idx] / data["Open"][0])
    return data


def Change_per(data):
    exData = data
    exData["Change%"] = np.nan

    for idx in range(1, len(data)):
        # print(idx)
        data.loc[data.index[idx], "Change%"] = (float(data["Close"][idx] / data["Close"][idx - 1]) - 1) * 100
    return data


def MA5(data):
    exData = data  # .head(10)
    exData["5MA"] = np.nan

    for index in range(5, len(exData)):
        list = []
        for i in range(5):
            list.append(float(exData["Close"][index - i]))
        exData["5MA"][index] = float(sum(list) / 5)
    return exData


def MA5Scale(data):
    data["5MAScale"] = np.nan
    for idx in range(0, len(data)):
        data.loc[data.index[idx], "5MAScale"] = float(data["5MA"][idx] / data["Open"][0])
    return data


def MA5Diff(data):
    data["5MADiff"] = np.nan
    for idx in range(0, len(data)):
        data.loc[data.index[idx], "5MADiff"] = float((data["Open"][idx] - data["5MA"][idx]) / data["5MA"][idx]) * 100
    return data


def Bias5(data):
    # NOTE: This function must be called after function "MovingAverage"
    exData = data  # .head(10)
    exData["5Bias"] = np.nan

    for index in range(5, len(data)):
        exData["5Bias"][index] = (float(exData["Close"][index]) - exData["5MA"][index]) / exData["5MA"][index] * 100
    return exData


def RSI(data):
    exData = data  # .head(30)
    exData["RSI"] = np.nan
    for i in range(14, len(exData)):
        list_up = []
        list_down = []
        for j in range(14):
            if float(exData["Close"][i - j]) > float(exData["Close"][i - j - 1]):
                list_up.append(float(exData["Close"][i - j]) - float(exData["Close"][i - j - 1]))
            elif float(exData["Close"][i - j]) < float(exData["Close"][i - j - 1]):
                list_down.append(float(exData["Close"][i - j - 1]) - float(exData["Close"][i - j]))
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


def WR(data):  # Used parameters: Price, High, Low
    exData = data  # .head(10)
    exData["WR"] = np.nan
    for i in range(10, len(exData)):
        list_high = []
        list_low = []
        for j in range(10):
            list_high.append(float(exData["High"][i - j]))
            list_low.append(float(exData["Low"][i - j]))
        if (max(list_high) - min(list_low)) == 0:
            exData["WR"][i] = -50
        else:
            exData["WR"][i] = ((max(list_high) - float(exData["Close"][i])) / (max(list_high) - min(list_low))) * -100
    return exData


def CreateTI(a):
    aa = OpenScale(a)
    b = Change_per(aa)
    c = MA5(b)
    cc = MA5Scale(c)
    ccc = MA5Diff(cc)
    d = Bias5(ccc)
    f = RSI(d)
    e = WR(f)
    return e


def DataPreprocessing(path):
    for file in glob.glob(os.path.join(path, '*.csv')):
        df = pd.read_csv(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        if filename == 'Celltrion':
            print("Filling missing value in Celltrion.csv")
            # 2013년 03월 22일 이전 가격 데이터를 2배, 거래량을 1/2배
            mask = (df['Date'] >= '2010-01-04') & (df['Date'] <= '2013-02-28')
            df.loc[mask, 'Open'] = df.loc[mask, 'Open'].apply(multiply_two)
            df.loc[mask, 'High'] = df.loc[mask, 'High'].apply(multiply_two)
            df.loc[mask, 'Low'] = df.loc[mask, 'Low'].apply(multiply_two)
            df.loc[mask, 'Close'] = df.loc[mask, 'Close'].apply(multiply_two)
            df.loc[mask, 'Volume'] = df.loc[mask, 'Volume'].apply(divide_two)

            # 2013년 03월 04일 ~ 03월 21일 거래량을 2013년 02월 28일 거래량으로 맞춤
            mask = (df['Date'] >= '2013-03-04') & (df['Date'] <= '2013-03-21')
            df.loc[mask, 'Volume'] = df[df['Date'] == '2013-02-28']['Volume'].values[0]
            # print(df.loc[mask])

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)

            df = CreateTI(df)

            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'Celltrion_P.csv', index=False)

        elif filename == 'HyundaiMotor':

            print("Filling missing value in HyndaiMotor.csv")
            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'HyundaiMotor_P.csv', index=False)

        elif filename == 'NAVER':
            print("Filling missing value in NAVER.csv")
            # 2013년 07월 29일 이전 가격 데이터를 1.5862배
            mask = (df['Date'] >= '2010-01-04') & (df['Date'] <= '2013-07-29')
            df.loc[mask, 'Open'] = df.loc[mask, 'Open'].apply(multiply_1_5862)
            df.loc[mask, 'High'] = df.loc[mask, 'High'].apply(multiply_1_5862)
            df.loc[mask, 'Low'] = df.loc[mask, 'Low'].apply(multiply_1_5862)
            df.loc[mask, 'Close'] = df.loc[mask, 'Close'].apply(multiply_1_5862)
            df.loc[mask, 'Volume'] = df.loc[mask, 'Volume'].apply(divide_1_5862)

            # 2013년 07월 30일 ~ 08월 28일 거래량을 2013년 07월 29일 데이터로 맞춤
            mask = (df['Date'] >= '2013-07-30') & (df['Date'] <= '2013-08-28')
            df.loc[mask, 'Open'] = df[df['Date'] == '2013-07-29']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2013-07-29']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2013-07-29']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2013-07-29']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2013-07-29']['Volume'].values[0]

            # 2018년 10월 05일 이전 가격 데이터를 0.2배
            mask = (df['Date'] >= '2010-01-04') & (df['Date'] <= '2018-10-05')
            df.loc[mask, 'Open'] = df.loc[mask, 'Open'].apply(multiply_0_2)
            df.loc[mask, 'High'] = df.loc[mask, 'High'].apply(multiply_0_2)
            df.loc[mask, 'Low'] = df.loc[mask, 'Low'].apply(multiply_0_2)
            df.loc[mask, 'Close'] = df.loc[mask, 'Close'].apply(multiply_0_2)
            df.loc[mask, 'Volume'] = df.loc[mask, 'Volume'].apply(divide_0_2)

            # 2018년 10월 08일 ~ 10월 11일 거래량을 2018년 10월 05일 데이터로 맞춤
            mask = (df['Date'] >= '2018-10-08') & (df['Date'] <= '2018-10-11')
            df.loc[mask, 'Open'] = df[df['Date'] == '2018-10-05']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2018-10-05']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2018-10-05']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2018-10-05']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2018-10-05']['Volume'].values[0]

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'NAVER_P.csv', index=False)

        elif filename == 'Kakao':
            print("Filling missing value in KaKao.csv")

            # 2014년 05월 26일 데이터를 05월 23일 데이터로 맞춤
            mask = (df['Date'] == '2014-05-26')
            df.loc[mask, 'Open'] = df[df['Date'] == '2014-05-23']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2014-05-23']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2014-05-23']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2014-05-23']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2014-05-23']['Volume'].values[0]

            # 2021년 04월 09일 이전 가격 데이터를 0.2배
            mask = (df['Date'] >= '2010-01-04') & (df['Date'] <= '2021-04-09')
            df.loc[mask, 'Open'] = df.loc[mask, 'Open'].apply(multiply_0_2)
            df.loc[mask, 'High'] = df.loc[mask, 'High'].apply(multiply_0_2)
            df.loc[mask, 'Low'] = df.loc[mask, 'Low'].apply(multiply_0_2)
            df.loc[mask, 'Close'] = df.loc[mask, 'Close'].apply(multiply_0_2)
            df.loc[mask, 'Volume'] = df.loc[mask, 'Volume'].apply(divide_0_2)

            # 2021년 04월 12일 ~ 04월 14일 데이터를 04월 09일 데이터로 맞춤
            mask = (df['Date'] >= '2021-04-12') & (df['Date'] <= '2021-04-14')
            df.loc[mask, 'Open'] = df[df['Date'] == '2021-04-09']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2021-04-09']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2021-04-09']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2021-04-09']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2021-04-09']['Volume'].values[0]

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df.drop_duplicates(keep='last', inplace=True)
            df = df.reset_index(drop=True)

            df = CreateTI(df)

            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'Kakao_P.csv', index=False)

        elif filename == "LGChemical":
            print("Filling missing value in LGChemical.csv")

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'LGChemical_P.csv', index=False)

        elif filename == "LGH_H":
            print("Filling missing value in LGH_H.csv")
            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'LGH_H_P.csv', index=False)

        elif filename == "SamsungBiologics":
            print("Filling missing value in SamsungBiologics.csv")

            # 2018년 11월 15일 ~ 12월 10일 데이터를 11월 14일 데이터로 맞춤
            mask = (df['Date'] >= '2018-11-15') & (df['Date'] <= '2018-12-10')
            df.loc[mask, 'Open'] = df[df['Date'] == '2018-11-14']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2018-11-14']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2018-11-14']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2018-11-14']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2018-11-14']['Volume'].values[0]

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'SamsungBiologics_P.csv', index=False)

        elif filename == "SamsungElectronics":
            print("Filling missing value in SamsungElectronics.csv")

            # 2018년 4월 30일 ~ 05월 03일 데이터를 04월 27일 데이터로 맞춤
            mask = (df['Date'] >= '2018-04-30') & (df['Date'] <= '2018-05-03')
            df.loc[mask, 'Open'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2018-04-27']['Volume'].values[0]

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'SamsungElectronics_P.csv', index=False)

        elif filename == "SamsungElectronics2":
            print("Filling missing value in SamsungElectronics2.csv")

            # 2018년 4월 30일 ~ 05월 03일 데이터를 04월 27일 데이터로 맞춤
            mask = (df['Date'] >= '2018-04-30') & (df['Date'] <= '2018-05-03')
            df.loc[mask, 'Open'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'High'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Low'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Close'] = df[df['Date'] == '2018-04-27']['Close'].values[0]
            df.loc[mask, 'Volume'] = df[df['Date'] == '2018-04-27']['Volume'].values[0]

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'SamsungElectronics2_P.csv', index=False)

        elif filename == "SamsungSDI":
            print("Filling missing value in SamsungSDI.csv")

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'SamsungSDI_P.csv', index=False)

        elif filename == 'SKhynix':
            print("Filling missing value in SKhynix.csv")

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'SKhynix_P.csv', index=False)

        elif filename == 'KOSPI':
            print("Filling missing value in KOSPI.csv")

            # 2015월 08월 14일, 2017년 09월 26일 행 제거
            df.drop(df[df['Date'] == '2015-08-14'].index, inplace=True)
            df.drop(df[df['Date'] == '2017-09-26'].index, inplace=True)
            df.drop_duplicates(keep='last', inplace=True)

            df['Volume'] = df['Volume'].str.replace('\D+', '').astype(int)

            df = df.reset_index(drop=True)
            df = CreateTI(df)
            df.drop(df.index[[v for v in range(14)]], axis=0, inplace=True)
            df = df.reset_index(drop=True)

            df.to_csv(path + 'KOSPI_P.csv', index=False)

        else:
            pass
