from sklearn import svm
import pickle
import DataGenerator


def main():
    test_day = ['2021-05-08','2021-05-09','2021-05-15', '2021-05-16']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')
    model = svm.SVC(C=0.7,kernel='poly',random_state=0,probability=True,class_weight='balanced')
    model.fit(training_x, training_y)

    print('completed training model')

    # TODO: fix pickle file name
    filename = 'team03_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('save complete')


if __name__ == '__main__':
    main()

