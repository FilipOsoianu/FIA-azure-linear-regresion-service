import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(filePath):
    headers = [
        "IDK1",
        "IDK2",
        "ComplexAge",
        "TotalRooms",
        "TotalBedrooms",
        "ComplexInhabitants",
        "ApartamentsNumber",
        "IDK3",
        "MedianComplexValue"
    ]
    return pd.read_csv(filePath, names=headers)


def parce_data(data):
    Y = data.iloc[:, 8].values
    X = data.drop(['MedianComplexValue'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    return X_train, Y_train, X_test, Y_test


def liniar_regresion(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    accuracy = model.score(X_train, Y_train)
    return accuracy


def predictt(model, array, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model.predict(array)
