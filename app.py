pip install Flask joblib boto3 scikit-learn numpy
from flask import Flask, request, jsonify
import joblib
import numpy as np
import boto3

app = Flask(__name__)

# Download the model and scaler from S3
s3 = boto3.client('s3')
bucket_name = 'rkr-info.xyz'
model_filename = 'model.pkl'
scaler_filename = 'scaler.pkl'
s3.download_file(bucket_name, model_filename, model_filename)
s3.download_file(bucket_name, scaler_filename, scaler_filename)

# Load the model and scaler
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict class and probability
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)

    # Respond with prediction and confidence
    response = {
        'prediction': int(prediction[0]),
        'confidence': probability[0].tolist()  # Probability for each class
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
