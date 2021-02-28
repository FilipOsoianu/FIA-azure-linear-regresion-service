"""
This script runs the python_webapp_flask application using a development server.
"""
from sklearn.linear_model import LinearRegression
from utils import get_data, parce_data, predictt
from flask import request, jsonify
from os import environ
from python_webapp_flask import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    
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

    app.run(HOST, PORT)
