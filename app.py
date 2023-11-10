from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import requests
app = Flask(__name__)

CORS(app)

def preprocessing(df):
    categorical_variables = ['brand', 'origin', 'type', 'gearbox', 'fuel', 'color', 'condition']
    numerical_variables = ['mileage_v2', 'manufacture_date', 'seats']
    label_encoder = pickle.load(open('./model_ckp/label_encoder.pkl','rb'))
    scaler = pickle.load(open('./model_ckp/scaler.pkl','rb'))
    for cat in categorical_variables:
        df[cat] = label_encoder.fit_transform(df[cat])
    df[numerical_variables] = scaler.transform(df[numerical_variables])
    return df

@app.route('/send-data/<int:n>', methods=['GET'])
def send_json(n):
    n = int(n)
    df = pd.read_csv("./data/processed_car.csv")
    data_to_send = df.iloc[n].to_dict()
    # Make a POST request to the receiver route
    receiver_url = 'http://localhost:5000/receive-json'  # Update with your actual URL
    response = requests.post(receiver_url, json=data_to_send)

    # Print the response from the receiver route
    print("Response from receiver route:", response.json())

    return jsonify(data_to_send)

# Route to receive JSON data
@app.route('/receive-json', methods=['POST'])
def receive_json():
    received_data = request.get_json()

    # You can now work with the received JSON data
    # For example, print it to the console
    # print("Received JSON data:", received_data)

    # You can also send a response back if needed
    response_data = {'message': 'JSON data received successfully'}
    df = pd.DataFrame(received_data, index=[0])
    y = df['price'].values
    x = df.drop(columns=['price'])
    x = preprocessing(x)
    model = pickle.load(open('./model_ckp/xgboostmodel.pkl','rb'))
    y_hat = np.abs(model.predict(x))
    print("----------------------------------------------------------------")
    print(f"True price: {y}, Predicted price: {y_hat}")
    print("----------------------------------------------------------------")
    return jsonify(response_data)
if __name__ == '__main__':
    app.run()