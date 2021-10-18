import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import _pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_size = int(len(data)*ratio)
    test_size1 = shuffled[:test_size]
    train_size = shuffled[test_size:]
    return data.iloc[train_size], data.iloc[test_size1]


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    train, test = data_split(df, 0.2)

    X_train = train[['fever', 'body', 'age',
                     'runnynoice', 'breath']].to_numpy()
    X_test = test[['fever', 'body', 'age', 'runnynoice', 'breath']].to_numpy()
    y_train = train[['probality']].to_numpy().reshape(2060,)
    y_test = test[['probality']].to_numpy().reshape(514,)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    file = open('model.pkl', 'wb')
    _pickle.dump(clf, file)
    file.close()
