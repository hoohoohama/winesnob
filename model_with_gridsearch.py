import os
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main():
    # Load red wine data.
    data = pd.read_csv('./datasets/winequality-red.csv', sep=';', quotechar='"')

    # print the first 5 rows
    # print(data.head())
    print(data.shape)
    # display summary statistics
    # print(data.describe())

    # separate target from training features
    y = data.quality
    X = data.drop('quality', axis=1)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=y)

    # Declare data preprocessing steps
    pipeline = make_pipeline(preprocessing.StandardScaler(),
                             RandomForestRegressor(n_estimators=100))

    # List some parameters that we want to tune
    hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                       'randomforestregressor__max_depth': [None, 5, 3, 1]}

    # Tune model using cross-validation pipeline
    clf = GridSearchCV(pipeline, hyperparameters, cv=10)

    # Fit and tune model
    clf.fit(X_train, y_train)

    print(clf.best_params_)

    # Evaluate model pipeline on test data
    # make a prediction
    y_pred = clf.predict(X_test)

    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))

    # save model to a .pkl file
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(clf, './output/rf_regressor.pkl')

    # load model again from .pkl file
    clf2 = joblib.load('./output/rf_regressor.pkl')

    # Predict data set using loaded model
    y_pred = clf2.predict(X_test)
    print(r2_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    # execute only if run as a script
    main()