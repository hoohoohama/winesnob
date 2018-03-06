import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split


def is_tasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


def main():
    # load dataset
    df = pd.read_csv('winequality-red.csv', sep=';', quotechar='"')

    # perform some data transformation
    df['tasty'] = df['quality'].apply(is_tasty)
    data = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
    target = df['tasty']

    # split dataset into training and test sets
    #
    # we'll create training and testing subsets that we'll use to train our classifiers.It is best practice to always
    # try train and test your classifiers on different datasets.Here we'll take one-third of the original population
    # and use it for testing, and the other two-thirds will be used for training the classifiers. Note that we can
    # specify a random_state seed in order to get the same results for the same input data if we want to replicate
    # this experiement later.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=123)

    # tree = DecisionTreeClassifier(max_depth=5)
    # tree.fit(data_train, target_train)

    # # use RandomForestClassifier
    tree = RandomForestClassifier(max_depth=None,
                                  n_estimators=100,
                                  max_features='auto')
    tree.fit(data_train, target_train)

    # List some parameters that we want to tune
    # hyperparameters = {class_weight=None, criterion='gini',
    #         max_depth=5, max_features='auto', max_leaf_nodes=None,
    #         min_impurity_decrease=0.0, min_impurity_split=None,
    #         min_samples_leaf=1, min_samples_split=2,
    #         min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
    #         oob_score=False, random_state=None, verbose=0,
    #         warm_start=False

    # # use GradientBoostingClassifier
    # tree = GradientBoostingClassifier(max_depth=5)
    # tree.fit(data_train, target_train)

    # check our results
    performance = precision_recall_fscore_support(target_test, tree.predict(data_test))
    print(performance)

    # Precision, Recall, Fscore, and Support
    print('Precision: ', performance[0])
    print('Recall: ', performance[1])
    print('Fscore: ', performance[2])
    print('Support: ', performance[3], '\n')

    print('Confusion Matrix:')
    print(confusion_matrix(target_test, tree.predict(data_test)), '\n')

    y_pred = tree.predict(data_test)
    print('r2_score: {}'.format(r2_score(target_test, y_pred)))
    print('mean_squared_error: {}'.format(mean_squared_error(target_test, y_pred)))

    # save model to a .pkl file
    joblib.dump(tree, 'model.pkl')

    # load model again from .pkl file
    tree2 = joblib.load('model.pkl')

    y_pred = tree2.predict(data_test)
    print('r2_score: {}'.format(r2_score(target_test, y_pred)))
    print('mean_squared_error: {}'.format(mean_squared_error(target_test, y_pred)))

    print('Confusion Matrix:')
    print(confusion_matrix(target_test, tree2.predict(data_test)), '\n')


if __name__ == '__main__':
    # execute only if run as a script
    main()
