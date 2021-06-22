import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings(action='ignore')

DATA_PATH = 'D:/2021/학교/1학기/데이터기반학습/LFD_Project3/data/'

column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'weather', 'rank', 'lane', 'horse', 'home',
                    'gender', 'age', 'weight', 'rating', 'jockey', 'trainer', 'owner', 'single_odds', 'double_odds'],
    'jockey': ['jockey', 'j_group', 'j_birth', 'j_age', 'j_debut', 'j_weight', 'j_weight_2', 'j_race_count', 'j_first',
               'j_second',
               'j_1yr_count', 'j_1yr_first', 'j_1yr_second'],
    'owner': ['owner', 'o_reg_horse', 'o_unreg_horse', 'o_owned_horse', 'o_reg_date', 'o_1yr_count', 'o_1yr_first',
              'o_1yr_second',
              'o_1yr_third', 'o_1yr_money', 'o_race_count', 'o_first', 'o_second', 'o_third', 'o_owner_money'],
    'trainer': ['trainer', 't_group', 't_birth', 't_age', 't_debut', 't_race_count', 't_first', 't_second',
                't_1yr_count', 't_1yr_first',
                't_1yr_second'],
    'horse': ['horse', 'h_home', 'h_gender', 'h_birth', 'h_age', 'h_class', 'h_group', 'h_trainer', 'h_owner',
              'h_father', 'h_mother',
              'h_race_count', 'h_first', 'h_second', 'h_1yr_count', 'h_1yr_first', 'h_1yr_second', 'h_horse_money',
              'h_rating',
              'h_price'],
}

# TODO: select columns to use
used_column_name = {
    'race_result': ['date', 'race_num', 'lane', 'jockey', 'horse', 'gender', 'home', 'rank', 'trainer', 'owner',
                    'weather', 'track_length'],
    'jockey': ['date', 'jockey', 'j_weight', 'j_race_count', 'j_first', 'j_second', 'j_1yr_count', 'j_1yr_first',
               'j_1yr_second', 'j_debut'],
    'owner': ['date', 'owner', 'o_owner_money', 'o_race_count', 'o_first', 'o_second', 'o_third', 'o_1yr_count',
              'o_1yr_first', 'o_1yr_second', 'o_1yr_third'],
    'trainer': ['date', 'trainer', 't_race_count', 't_first', 't_second', 't_1yr_count', 't_1yr_first', 't_1yr_second',
                't_debut', 't_age'],
    'horse': ['date', 'horse', 'h_first', 'h_second', 'h_1yr_first', 'h_1yr_second', 'h_class', 'h_rating',
              'h_race_count', 'h_1yr_count'],
}

column_to_scale = ["track_length", "h_rating", "j_weight", "j_debut_year", "o_owner_money", "t_debut_year"]


def load_data():
    df_dict = dict()  # key: data type(e.g. jockey, trainer, ...), value: corresponding dataframe

    for data_type in ['horse', 'jockey', 'owner', 'trainer', 'race_result']:
        fnames = sorted(os.listdir(DATA_PATH + data_type))
        df = pd.DataFrame()

        # concatenate all text files in the directory
        for fname in fnames:
            tmp = pd.read_csv(os.path.join(DATA_PATH, data_type, fname), header=None, sep=",", encoding='cp949',
                              names=column_name[data_type])
            if data_type == 'race_result':
                tmp['owner'] = tmp['owner'].str.strip().replace(r'♠', '', regex=True)

            if data_type != 'race_result':
                date = fname.split('.')[0]
                tmp['date'] = date[:4] + "-" + date[4:6] + "-" + date[-2:]

            df = pd.concat([df, tmp])

        # cast date column to dtype datetime
        df['date'] = df['date'].astype('datetime64[ns]')

        # append date offset to synchronize date with date of race_result data
        if data_type != 'race_result':
            df1 = df.copy()
            df1['date'] += pd.DateOffset(days=2)  # saturday
            df2 = df.copy()
            df2['date'] += pd.DateOffset(days=3)  # sunday
            df3 = df.copy()
            df3['date'] += pd.DateOffset(days=1)  # friday
            df = df1.append(df2)
            df = df.append(df3)

        # select columns to use
        df = df[used_column_name[data_type]]

        # insert dataframe to dictionary
        df_dict[data_type] = df

    ####### DO NOT CHANGE #######

    df_dict['race_result']['rank'].replace('1', 1., inplace=True)
    df_dict['race_result']['rank'].replace('2', 2., inplace=True)
    df_dict['race_result']['rank'].replace('3', 3., inplace=True)
    df_dict['race_result']['rank'].replace('4', 4., inplace=True)
    df_dict['race_result']['rank'].replace('5', 5., inplace=True)
    df_dict['race_result']['rank'].replace('6', 6., inplace=True)
    df_dict['race_result']['rank'].replace('7', 7., inplace=True)
    df_dict['race_result']['rank'].replace('8', 8., inplace=True)
    df_dict['race_result']['rank'].replace('9', 9., inplace=True)
    df_dict['race_result']['rank'].replace('10', 10., inplace=True)
    df_dict['race_result']['rank'].replace('11', 11., inplace=True)
    df_dict['race_result']['rank'].replace('12', 12., inplace=True)
    df_dict['race_result']['rank'].replace('13', 13., inplace=True)
    df_dict['race_result']['rank'].replace(' ', np.nan, inplace=True)

    # drop rows with rank missing values
    df_dict['race_result'].dropna(subset=['rank'], inplace=True)
    df_dict['race_result']['rank'] = df_dict['race_result']['rank'].astype('int')

    # make a column 'win' that indicates whether a horse ranked within the 3rd place
    df_dict['race_result']['win'] = df_dict['race_result'].apply(lambda x: 1 if x['rank'] < 4 else 0, axis=1)

    #################################

    # TODO: Make Features

    # Gender Dummy
    df_dict['race_result']['gender'].replace({'암': 'F', '수': 'M', '거': 'N'}, inplace=True)
    df_dict['race_result'] = pd.concat([df_dict['race_result'], pd.get_dummies(df_dict['race_result']['gender'])],
                                       axis=1)

    # Race Weather Dummy
    df_dict['race_result']['weather'].replace({'맑음': 'G', '흐림': 'B', '눈': 'S', '비': 'R', '강풍': 'W', '안개': 'A'},
                                              inplace=True)
    df_dict['race_result'] = pd.concat([df_dict['race_result'], pd.get_dummies(df_dict['race_result']['weather'])],
                                       axis=1)
    del df_dict['race_result']['weather']

    # Horse Home Dummy
    df_dict['race_result']['home'].replace(
        {'한': 'Ko', '한(포)': 'Ko', '미': 'Fo', '호': 'Fo', '일': 'Fo', '뉴': 'Fo', '캐': 'Fo', '아일': 'Fo', '프': 'Fo',
         '영': 'Fo', '모': 'Fo', '남': 'Fo', '브': 'Fo'}, inplace=True)
    df_dict['race_result'] = pd.concat([df_dict['race_result'], pd.get_dummies(df_dict['race_result']['home'])], axis=1)

    # NaN Rating to Zero
    df_dict['horse']['h_rating'].replace(r' ', np.nan, regex=True, inplace=True)
    df_dict['horse']['h_rating'] = pd.to_numeric(df_dict['horse']['h_rating'], downcast='integer')
    df_dict['horse']['h_rating'] = df_dict['horse']['h_rating'].fillna(0)  # No rating into 0 value

    # 외미 Rating to 35
    df_dict['horse'].loc[df_dict['horse']['h_class'] == '외미', 'h_rating'] = 35

    # j_debut, t_debut type to datetime
    df_dict['jockey']['j_debut'] = df_dict['jockey']['j_debut'].astype('datetime64[ns]')

    #     df_dict['trainer']['t_debut'] = df_dict['trainer']['t_debut'].astype('datetime64[ns]')

    df_dict['horse']['h_1yr_1strate'] = df_dict['horse']['h_1yr_first'] / df_dict['horse']['h_1yr_count']
    df_dict['horse']['h_1yr_2ndrate'] = df_dict['horse']['h_1yr_second'] / df_dict['horse']['h_1yr_count']
    df_dict['horse']['h_1strate'] = df_dict['horse']['h_first'] / df_dict['horse']['h_race_count']
    df_dict['horse']['h_2ndrate'] = df_dict['horse']['h_second'] / df_dict['horse']['h_race_count']

    df_dict['jockey']['j_1yr_1strate'] = df_dict['jockey']['j_1yr_first'] / df_dict['jockey']['j_1yr_count']
    df_dict['jockey']['j_1yr_2ndrate'] = df_dict['jockey']['j_1yr_second'] / df_dict['jockey']['j_1yr_count']
    df_dict['jockey']['j_1strate'] = df_dict['jockey']['j_first'] / df_dict['jockey']['j_race_count']
    df_dict['jockey']['j_2ndrate'] = df_dict['jockey']['j_second'] / df_dict['jockey']['j_race_count']

    df_dict['trainer']['t_1yr_1strate'] = df_dict['trainer']['t_1yr_first'] / df_dict['trainer']['t_1yr_count']
    df_dict['trainer']['t_1yr_2ndrate'] = df_dict['trainer']['t_1yr_second'] / df_dict['trainer']['t_1yr_count']
    df_dict['trainer']['t_1strate'] = df_dict['trainer']['t_first'] / df_dict['trainer']['t_race_count']
    df_dict['trainer']['t_2ndrate'] = df_dict['trainer']['t_second'] / df_dict['trainer']['t_race_count']

    df_dict['owner']['o_1yr_1strate'] = df_dict['owner']['o_1yr_first'] / df_dict['owner']['o_1yr_count']
    df_dict['owner']['o_1yr_2ndrate'] = df_dict['owner']['o_1yr_second'] / df_dict['owner']['o_1yr_count']
    df_dict['owner']['o_1yr_3rdrate'] = df_dict['owner']['o_1yr_third'] / df_dict['owner']['o_1yr_count']
    df_dict['owner']['o_1strate'] = df_dict['owner']['o_first'] / df_dict['owner']['o_race_count']
    df_dict['owner']['o_2ndrate'] = df_dict['owner']['o_second'] / df_dict['owner']['o_race_count']
    df_dict['owner']['o_3rdrate'] = df_dict['owner']['o_third'] / df_dict['owner']['o_race_count']

    ################################################## 추가
    df_dict['horse']['h_1yr_rate'] = (df_dict['horse']['h_1yr_first'] + df_dict['horse']['h_1yr_second']) / \
                                     df_dict['horse']['h_1yr_count']
    df_dict['horse']['h_total_rate'] = (df_dict['horse']['h_first'] + df_dict['horse']['h_second']) / df_dict['horse'][
        'h_race_count']
    df_dict['jockey']['j_1yr_rate'] = (df_dict['jockey']['j_1yr_first'] + df_dict['jockey']['j_1yr_second']) / \
                                      df_dict['jockey']['j_1yr_count']
    df_dict['jockey']['j_total_rate'] = (df_dict['jockey']['j_first'] + df_dict['jockey']['j_second']) / \
                                        df_dict['jockey']['j_race_count']
    df_dict['trainer']['t_1yr_rate'] = (df_dict['trainer']['t_1yr_first'] + df_dict['trainer']['t_1yr_second']) / \
                                       df_dict['trainer']['t_1yr_count']
    df_dict['trainer']['t_total_rate'] = (df_dict['trainer']['t_first'] + df_dict['trainer']['t_second']) / \
                                         df_dict['trainer']['t_race_count']
    df_dict['owner']['o_1yr_rate'] = (df_dict['owner']['o_1yr_first'] + df_dict['owner']['o_1yr_second'] +
                                      df_dict['owner']['o_1yr_third']) / df_dict['owner']['o_1yr_count']
    df_dict['owner']['o_total_rate'] = (df_dict['owner']['o_first'] + df_dict['owner']['o_second'] + df_dict['owner'][
        'o_third']) / df_dict['owner']['o_race_count']

    df_dict['horse']['h_1yr_rate'].fillna(value=0, inplace=True)
    df_dict['horse']['h_total_rate'].fillna(value=0, inplace=True)
    df_dict['jockey']['j_1yr_rate'].fillna(value=0, inplace=True)
    df_dict['jockey']['j_total_rate'].fillna(value=0, inplace=True)
    df_dict['trainer']['t_1yr_rate'].fillna(value=0, inplace=True)
    df_dict['trainer']['t_total_rate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_1yr_rate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_total_rate'].fillna(value=0, inplace=True)
    ##################################################

    df_dict['horse']['h_1yr_1strate'].fillna(value=0, inplace=True)
    df_dict['horse']['h_1yr_2ndrate'].fillna(value=0, inplace=True)
    df_dict['horse']['h_1strate'].fillna(value=0, inplace=True)
    df_dict['horse']['h_2ndrate'].fillna(value=0, inplace=True)

    df_dict['trainer']['t_1yr_1strate'].fillna(value=0, inplace=True)
    df_dict['trainer']['t_1yr_2ndrate'].fillna(value=0, inplace=True)
    df_dict['trainer']['t_1strate'].fillna(value=0, inplace=True)
    df_dict['trainer']['t_2ndrate'].fillna(value=0, inplace=True)

    df_dict['jockey']['j_1yr_1strate'].fillna(value=0, inplace=True)
    df_dict['jockey']['j_1yr_2ndrate'].fillna(value=0, inplace=True)
    df_dict['jockey']['j_1strate'].fillna(value=0, inplace=True)
    df_dict['jockey']['j_2ndrate'].fillna(value=0, inplace=True)

    df_dict['owner']['o_1yr_1strate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_1yr_2ndrate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_1yr_3rdrate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_1strate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_2ndrate'].fillna(value=0, inplace=True)
    df_dict['owner']['o_3rdrate'].fillna(value=0, inplace=True)

    # drop duplicated rows
    df_dict['jockey'].drop_duplicates(subset=['date', 'jockey'], inplace=True)
    df_dict['owner'].drop_duplicates(subset=['date', 'owner'], inplace=True)
    df_dict['trainer'].drop_duplicates(subset=['date', 'trainer'], inplace=True)

    # merge dataframes
    df = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')
    df = df.merge(df_dict['jockey'], on=['date', 'jockey'], how='left')
    df = df.merge(df_dict['owner'], on=['date', 'owner'], how='left')
    df = df.merge(df_dict['trainer'], on=['date', 'trainer'], how='left')

    # drop unnecessary columns which are used only for merging dataframes
    df.drop(
        ['horse', 'jockey', 'trainer', 'owner', 'h_1yr_first', 'h_1yr_second', 'h_first', 'h_second', 'h_race_count',
         'h_1yr_count', 'j_1yr_first', 'j_1yr_second', 'j_first', 'j_second', 'j_race_count', 'j_1yr_count',
         't_1yr_first', 't_1yr_second', 't_first', 't_second', 't_race_count', 't_1yr_count', 'o_1yr_first',
         'o_1yr_second', 'o_1yr_third', 'o_first', 'o_second', 'o_third', 'o_race_count', 'o_1yr_count'], axis=1,
        inplace=True)

    df.loc[(df['h_class'] == '외1') & (df['h_rating'] == 0), 'h_rating'] = 81
    df.loc[(df['h_class'] == '외2') & (df['h_rating'] == 0), 'h_rating'] = 66
    df.loc[(df['h_class'] == '외3') & (df['h_rating'] == 0), 'h_rating'] = 51
    df.loc[(df['h_class'] == '외4') & (df['h_rating'] == 0), 'h_rating'] = 35
    df.loc[(df['h_class'] == '국1') & (df['h_rating'] == 0), 'h_rating'] = 81
    df.loc[(df['h_class'] == '국2') & (df['h_rating'] == 0), 'h_rating'] = 66
    df.loc[(df['h_class'] == '국3') & (df['h_rating'] == 0), 'h_rating'] = 51
    df.loc[(df['h_class'] == '국4') & (df['h_rating'] == 0), 'h_rating'] = 35
    df.loc[(df['h_class'] == '국5') & (df['h_rating'] == 0), 'h_rating'] = 10
    df.drop(['h_class'], axis=1, inplace=True)
    
    # category data preprocessing - Calibration
    df.loc[df['gender']=='F', 'gender'] = len(df.loc[(df['gender']=='F') & (df['win']==1)])/len(df.loc[df['gender']=='F'])
    df.loc[df['gender']=='M', 'gender'] = len(df.loc[(df['gender']=='M') & (df['win']==1)])/len(df.loc[df['gender']=='M'])
    df.loc[df['gender']=='N', 'gender'] = len(df.loc[(df['gender']=='N') & (df['win']==1)])/len(df.loc[df['gender']=='N'])
    df['gender'].fillna(len(df.loc[df['win']==1])/len(df.loc[(df['win']==1) & (df['win']==1)]))
    
    df.loc[df['home']=='Ko', 'home'] = len(df.loc[(df['home']=='Ko') & (df['win']==1)])/len(df.loc[df['home']=='Ko'])
    df.loc[df['home']=='Fo', 'home'] = len(df.loc[(df['home']=='Fo') & (df['win']==1)])/len(df.loc[df['home']=='Fo'])
    df['home'].fillna(len(df.loc[df['win']==1])/len(df.loc[(df['win']==1) & (df['win']==1)]))
    
    # Calculated debut year
    # df.dropna(subset = ['j_debut'], axis=0, inplace=True)
    # df['j_debut_year'] = df['date'].dt.year - df['j_debut'].dt.year

    # Calculate debut year
    df.loc[df['j_debut'] != np.nan, 'j_debut_year'] = df['date'].dt.year - df['j_debut'].dt.year
    df.loc[df['j_debut'] == np.nan, 'j_debut_year'] = df.loc[df['j_debut'] != np.nan, 'j_debut_year'].mean()

    df.drop(['j_debut'], axis=1, inplace=True)

    # Calculate trainer debut year, if age under 0, debut year == -1
    df.loc[df['t_age'] <= 0, 't_debut'] = pd.to_datetime(
        str(int(df.loc[df['t_age'] > 0, 't_debut'].astype('datetime64[ns]').dt.year.mean())) + "0101").date()
    df['t_debut_year'] = df['date'].dt.year - df['t_debut'].astype('datetime64[ns]').dt.year

    df.drop(['t_debut'], axis=1, inplace=True)
    df.drop(['t_age'], axis=1, inplace=True)

    # drop if track length is 1500 and above
    df.drop(df[df['track_length'] > 1400].index, inplace=True)

    fill_flt_list = ['h_1yr_1strate', 'h_1yr_2ndrate', 'h_1strate', 'h_2ndrate',
                     'j_1yr_1strate', 'j_1yr_2ndrate', 'j_1strate', 'j_2ndrate',
                     'o_1yr_1strate', 'o_1yr_2ndrate', 'o_1yr_3rdrate', 'o_1strate', 'o_2ndrate', 'o_3rdrate',
                     't_1yr_1strate', 't_1yr_2ndrate', 't_1strate', 't_2ndrate', 'h_1yr_rate', 'h_total_rate',
                     'j_1yr_rate', 'j_total_rate', 't_1yr_rate', 't_total_rate', 'o_1yr_rate', 'o_total_rate']
    fill_int_list = ['h_rating', 'j_weight', 'o_owner_money', 'j_debut_year', 't_debut_year']

    for col in fill_flt_list:
        is_NaN = df.isnull()
        row_has_NaN = is_NaN[is_NaN[col] == True][col]

        for i in row_has_NaN.index:
            df[col][i] = \
            df[(df['date'] == df['date'][i]) & (df['race_num'] == df['race_num'][i]) & (df[col] != np.nan)][col].mean()
            # df_copy=df[df['date']==df['date'][i]]
            # df_copy=df_copy[df['race_num']==df['race_num'][i]]
            # df_copy=df_copy[df[col]!=np.nan]
            # df[col][i]=df_copy[col].mean()

    for col in fill_int_list:
        is_NaN = df.isnull()
        row_has_NaN = is_NaN[is_NaN[col] == True][col]
        for i in row_has_NaN.index:
            df[col][i] = \
            df[(df['date'] == df['date'][i]) & (df['race_num'] == df['race_num'][i]) & (df[col] != np.nan)][col].mean()
            # df_copy=df[df['date']==df['date'][i]]
            # df_copy=df_copy[df['race_num']==df['race_num'][i]]
            # df_copy=df_copy[df[col]!=np.nan]
            # df[col][i]=df_copy[col].mean()
        df[col] = round(df[col])

    # df_copy=df[df['date']==df['date'][i]]
    # df_copy=df_copy[df['lane']==df['lane'][i]]
    # df_copy=df_copy[df[col]!=np.nan]
    # df[col][i]=df_copy[col].mean()

    # eliminate any missing data

    # df = df.dropna(axis = 0, how ='any')
    df = df.reset_index(drop=True)

    df.to_csv('df_final.csv',index=None)
    return df


def get_data(test_day, is_training):
    if os.path.exists('df_final.csv'):
        print('preprocessed data exists')
        data_set = pd.read_csv('df_final.csv')
    else:
        print('preprocessed data NOT exists')
        print('loading data')
        data_set = load_data()

    # select training and test data by test day
    # TODO : cleaning or filling missing value
    training_data = data_set[~data_set['date'].isin(test_day)].fillna(0)
    test_data = data_set[data_set['date'].isin(test_day)].fillna(0)

    # TODO : make your input feature columns

    # select training x and y
    training_y = training_data['win']
    training_x = training_data.drop(['win', 'date', 'race_num', 'rank', 'lane'], axis=1)

    # select test x and y
    test_y = test_data['win']
    test_x = test_data.drop(['win', 'rank', 'lane'], axis=1)

    for col in column_to_scale:
        std_scaler = StandardScaler()
        #null_values = training_x[col].isnull()
        # training_x.loc[~null_values, [col]] = std_scaler.fit_transform(training_x.loc[~null_values, [col]])
        training_x.loc[:, [col]] = std_scaler.fit_transform(training_x.loc[:, [col]])
        # test_x.loc[~null_values, [col]] = std_scaler.transform(test_x.loc[~null_values, [col]])
        test_x.loc[:, [col]] = std_scaler.transform(test_x.loc[:, [col]])

    ##########
    inspect_test_data(test_x, test_day)
    drop_list = ['h_1yr_1strate', 'h_1yr_2ndrate', 'h_1strate', 'h_2ndrate',
                 'j_1yr_1strate', 'j_1yr_2ndrate', 'j_1strate', 'j_2ndrate',
        'o_1yr_1strate', 'o_1yr_2ndrate', 'o_1yr_3rdrate', 'o_1strate', 'o_2ndrate', 'o_3rdrate',
                 't_1yr_1strate', 't_1yr_2ndrate', 't_1strate', 't_2ndrate', 't_debut_year',
                 'o_owner_money', 'j_debut_year','t_total_rate']
    training_x.drop(drop_list, axis=1, inplace=True)
    test_x.drop(drop_list, axis=1, inplace=True)

    return (training_x, training_y) if is_training else (test_x, test_y)


def inspect_test_data(test_x, test_days):
    """
    Do not fix this function
    """
    df = pd.DataFrame()

    for test_day in test_days:
        fname = os.path.join(DATA_PATH, 'race_result', test_day.replace('-', '') + '.csv')
        tmp = pd.read_csv(fname, header=None, sep=",",
                          encoding='cp949', names=column_name['race_result'])
        tmp.replace(' ', np.nan, inplace=True)
        tmp.dropna(subset=['rank'], inplace=True)

        df = pd.concat([df, tmp])

    # print(test_x.shape[0])
    # print(df.shape[0])

    assert test_x.shape[0] == df.shape[0], 'your test data is wrong!'


def main():
    get_data(['2021-05-08','2021-05-09','2021-05-15', '2021-05-16'], is_training=True)


if __name__ == '__main__':
    main()