import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

@app.route("/accuracy", methods=["GET"])
def accuracy():
    with open("model/accuracy.txt") as f:
        result = f.read()
    return result

if __name__ == "__main__":
    app.run(debug=True)