import pickle
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings

warnings.filterwarnings(action='ignore')


def main():
    test_day = ['2021-05-08', '2021-05-09', '2021-05-15', '2021-05-16']
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)

    # TODO: fix pickle file name
    filename = 'team03_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model.get_params())

    # ================================ predict result ========================================
    pred_y = []
    for date in test_day:
        for race_num in test_x[test_x['date'] == date]['race_num'].unique():
            race_results = test_x[test_x['date'] == date][test_x['race_num'] == race_num]
            race_results.drop(['date', 'race_num'], axis=1, inplace=True)
            tmp = model.predict_proba(race_results)[:, 1]
            top_3 = sorted(range(len(tmp)), key=lambda i: tmp[i])[-3:]

            for j in range(len(tmp)):
                if j in top_3:
                    tmp[j] = 1
                else:
                    tmp[j] = 0
            for k in tmp:
                pred_y.append(k)

    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))


if __name__ == '__main__':
    main()