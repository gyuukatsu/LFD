import os
import pandas as pd
import numpy as np
import Preprocess

symbol_dict = {'cell': 'Celltrion_P',
               'hmotor': 'HyundaiMotor_P',
               'naver': 'NAVER_P',
               'kakao': 'Kakao_P',
               'lgchem': 'LGChemical_P',
               'lghnh': 'LGH_H_P',
               'bio': 'SamsungBiologics_P',
               'samsung1': 'SamsungElectronics_P',
               'samsung2': 'SamsungElectronics2_P',
               'sdi': 'SamsungSDI_P',
               'sk': 'SKhynix_P',
               'kospi': 'KOSPI_P',}

PATH="D:/2021/학교/1학기/데이터기반학습/LFD_Project4/data"

def symbol_to_path(symbol, base_dir=PATH):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)

    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                              usecols=['Date', 'Open','OpenScale', 'High', 'Low', 'Close', 'Volume', 'Change%', '5MAScale','5MADiff', '5Bias', 'RSI','WR'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'OpenScale': symbol + '_openScale', 'High': symbol + '_high', 'Low': symbol + '_low',
                                          'Close': symbol + '_close', 'Volume': symbol + '_volume', 'Change%': symbol+'_change%',
                                          '5MAScale':symbol+'_5maScale', '5Bias':symbol+'_5bias', 'RSI':symbol+'_rsi','WR':symbol+'_wr',
                                          'Open': symbol + '_open', '5MADiff':symbol+'_5maDiff'})
        df = df.join(df_temp)

    # TODO: cleaning or filling missing value
    df = df.dropna()

    return df


def make_features(trade_company_list, start_date, end_date, is_training):
    Preprocess.DataPreprocessing(PATH+'/')
    # TODO: Choose symbols to make feature
    # symbol_list = ['Celltrion', 'HyundaiMotor', 'NAVER', 'Kakao', 'LGChemical', 'LGH&H',
    #                 'SamsungElectronics', 'SamsungElectronics2', 'SamsungSDI', 'SKhynix', 'KOSPI']
    feature_company_list = ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']
    symbol_list = [symbol_dict[c] for c in feature_company_list]

    table = merge_data(start_date, end_date, symbol_list)

    # DO NOT CHANGE
    test_days = 3
    open_prices = np.asarray(table[[symbol_dict[c]+'_open' for c in trade_company_list]])
    close_prices = np.asarray(table[[symbol_dict[c]+'_close' for c in trade_company_list]])
    

    # TODO: select columns to use
    data = dict()
    for c in feature_company_list:
        data[c, 'close'] = table[symbol_dict[c] + '_close']
        data[c, 'open'] = table[symbol_dict[c] + '_open']
        data[c, 'open_ema_5'] = table[symbol_dict[c] + '_open'].ewm(alpha=0.33).mean()
        data[c, 'open_ema_20'] = table[symbol_dict[c] + '_open'].ewm(alpha=0.008).mean()
        data[c, 'RSI'] = table[symbol_dict[c] + '_rsi']
        data[c, '5MADiff'] = table[symbol_dict[c] + '_5maDiff']
        data[c, 'WR'] = table[symbol_dict[c] + '_wr']
        data[c, '5Bias'] = table[symbol_dict[c] + '_5bias']
        data[c, 'Change%'] = table[symbol_dict[c] + '_change%']
        data[c, 'openScale'] = table[symbol_dict[c] + '_openScale']
        data[c, 'emaDiff'] = data[c, 'open_ema_5'] - data[c, 'open_ema_20']


    # TODO: make features
    input_days = 2

    features = list()
    
    cha=[]
    maDiff=[]
    bias=[]
    rsi=[]
    wr=[]
    emaDiff=[]
    for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            cha.extend(data[symbol, 'Change%'].tolist())
            maDiff.extend(data[symbol, '5MADiff'].tolist())
            bias.extend(data[symbol, '5Bias'].tolist())
            rsi.extend(data[symbol, 'RSI'].tolist())
            wr.extend(data[symbol, 'WR'].tolist())
            emaDiff.extend(data[symbol, 'emaDiff'].tolist())
    cha_mm=[min(cha),max(cha)]
    maDiff_mm=[min(maDiff),max(maDiff)]
    bias_mm=[min(bias),max(bias)]
    rsi_mm=[min(rsi),max(rsi)]
    wr_mm=[min(wr),max(wr)]
    emaDiff_mm=[min(emaDiff),max(emaDiff)]
    
    for a in range(data['kospi', 'close'].shape[0] - input_days):

        # kospi close price
        kospi_close_feature = data['kospi', 'close'][a:a + input_days]

        # stock close price : lgchem, samsung1
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, 'Change%'][a:a + input_days]
            tmps.append(tmp)
        cha_feature = np.concatenate(tmps, axis=0)
        cha_feature = (cha_feature-cha_mm[0])/(cha_mm[1]-cha_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, '5MADiff'][a:a + input_days]
            tmps.append(tmp)
        maDiff_feature = np.concatenate(tmps, axis=0)
        maDiff_feature = (maDiff_feature-maDiff_mm[0])/(maDiff_mm[1]-maDiff_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, '5Bias'][a:a + input_days]
            tmps.append(tmp)
        bias_feature = np.concatenate(tmps, axis=0)
        bias_feature = (bias_feature-bias_mm[0])/(bias_mm[1]-bias_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, 'RSI'][a:a + input_days]
            tmps.append(tmp)
        rsi_feature = np.concatenate(tmps, axis=0)
        rsi_feature = (rsi_feature-rsi_mm[0])/(rsi_mm[1]-rsi_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, 'WR'][a:a + input_days]
            tmps.append(tmp)
        wr_feature = np.concatenate(tmps, axis=0)
        wr_feature = (wr_feature-wr_mm[0])/(wr_mm[1]-wr_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, 'emaDiff'][a:a + input_days]
            tmps.append(tmp)
        emaDiff_feature = np.concatenate(tmps, axis=0)
        emaDiff_feature = (emaDiff_feature-emaDiff_mm[0])/(emaDiff_mm[1]-emaDiff_mm[0])
        
        tmps = list()
        for symbol in ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk','kospi']:
            tmp = data[symbol, 'openScale'][a:a + input_days]
            tmps.append(tmp)
        openScale_feature = np.concatenate(tmps, axis=0)
        
        features.append(np.concatenate([
                                        #openScale_feature,
                                        cha_feature,
                                        #bias_feature,
                                        rsi_feature,
                                        emaDiff_feature
                                        ], axis=0))

    if not is_training:
        return open_prices[-test_days:], close_prices[-test_days:], features[-test_days:]

    return open_prices[input_days:], close_prices[input_days:], features


if __name__ == "__main__":
    trade_company_list = ['cell','hmotor','naver','kakao','lghnh','lgchem','samsung1','samsung2','sdi','sk']
    open, close, feature = make_features(trade_company_list, '2010-01-01', '2019-12-31', False)
    print(open,'\n')
    print(close,'\n')
    print(*feature[0],sep=' / ')