from flask import Flask, request, jsonify
import joblib
import numpy as np
import boto3
import os

app = Flask(__name__)

# Download the model from S3
s3 = boto3.client('s3')
bucket_name = 'rkr-info.xyz'
model_filename = 'model.pkl'
s3.download_file(bucket_name, model_filename, model_filename)

# Load the model
model = joblib.load(model_filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
