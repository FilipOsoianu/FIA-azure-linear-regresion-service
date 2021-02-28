"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request, jsonify
from python_webapp_flask import app
from .utils import get_data, parce_data, predictt
from sklearn.linear_model import LinearRegression


@app.route('/')
@app.route('/predict', methods=['GET'])
def predict():
    data = get_data('apartmentComplexData.csv')
    model = LinearRegression()
    X_train, Y_train, X_test, Y_test = parce_data(data)
    predict_array = []
    request_data = request.get_json()
    for req in request_data:
        predict_array.append([
            float(req["IDK1"]),
            float(req["IDK2"]),
            float(req["ComplexAge"]),
            float(req["TotalRooms"]),
            float(req["TotalBedrooms"]),
            float(req["ComplexInhabitants"]),
            float(req["ApartamentsNumber"]),
            float(req["IDK3"]),
        ])
    predictions = predictt(model, predict_array, X_train, Y_train)
    response = {}
    for i in range(len(predictions)):
        response['MedianComplexValue ' + str(i)] = predictions[i]
    return jsonify(response)
