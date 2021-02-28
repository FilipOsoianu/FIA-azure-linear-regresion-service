import pytest
from utils import get_data, parce_data, liniar_regresion
from sklearn.linear_model import LinearRegression


def test_accuracy():
    data = get_data('apartmentComplexData.csv')
    model = LinearRegression()
    X_train, Y_train, X_test, Y_test = parce_data(data)
    accuracy = liniar_regresion(model, X_train, Y_train, X_test, Y_test)

    assert accuracy > 0.60