from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load all models
with open("adaboost_breast_cancer.pkl", "rb") as f:
    breast_model = pickle.load(f)

with open("adaboost_diabetes.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("adaboost_heart_disease.pkl", "rb") as f:
    heart_model = pickle.load(f)

# Define features for each model
FEATURES = {
    "breast": ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"],
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "age", "sex", "chest pain type", "resting blood pressure",
        "serum cholestoral", "fasting blood sugar",
        "resting electrocardiographic results", "max heart rate",
        "exercise induced angina", "oldpeak", "ST segment",
        "major vessels", "thal"
    ]
}

MODELS = {
    "breast": breast_model,
    "diabetes": diabetes_model,
    "heart": heart_model
}

@app.route("/")
def home():
    return "Multi-Disease Prediction API is running!"

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in MODELS:
        return jsonify({"error": f"Invalid disease type: {disease}"}), 400

    data = request.get_json()

    try:
        features = FEATURES[disease]
        model = MODELS[disease]
        values = [float(data[feature]) for feature in features]
        input_array = np.array(values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        if disease == "breast":
            result = "Malignant" if prediction == 1 else "Benign"
        elif disease == "diabetes":
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        else:  # heart
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return jsonify({"prediction": result})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
# This code defines a Flask application that serves as an API for predicting multiple diseases using pre-trained AdaBoost models.