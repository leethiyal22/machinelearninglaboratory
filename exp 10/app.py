"""
app.py - Flask Web Application for Diabetes Prediction System Using Machine Learning
Educational and machine learning demonstration purposes only.
Not for medical diagnosis.
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash

app = Flask(__name__)
app.secret_key = 'diabetes-ml-demo-secret-key-local-only'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

# Global variables for model and metadata
model = None
metadata = {}

def load_resources():
    """Load model pipeline and dataset metadata."""
    global model, metadata
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        print("Model or metadata not found. Triggering train_model.py...")
        from train_model import train_and_evaluate
        train_and_evaluate()

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded model successfully. Selected Model: {metadata.get('selected_model', {}).get('name')}")

load_resources()

# Valid clinical feature ranges for input validation
FEATURE_SPECS = {
    'Pregnancies': {'type': int, 'min': 0, 'max': 25, 'label': 'Pregnancies', 'unit': 'count'},
    'Glucose': {'type': float, 'min': 0.0, 'max': 400.0, 'label': 'Glucose Level', 'unit': 'mg/dL'},
    'BloodPressure': {'type': float, 'min': 0.0, 'max': 240.0, 'label': 'Blood Pressure', 'unit': 'mm Hg'},
    'SkinThickness': {'type': float, 'min': 0.0, 'max': 110.0, 'label': 'Skin Thickness', 'unit': 'mm'},
    'Insulin': {'type': float, 'min': 0.0, 'max': 900.0, 'label': 'Insulin Level', 'unit': 'μU/mL'},
    'BMI': {'type': float, 'min': 0.0, 'max': 80.0, 'label': 'BMI', 'unit': 'kg/m²'},
    'DiabetesPedigreeFunction': {'type': float, 'min': 0.01, 'max': 3.5, 'label': 'Diabetes Pedigree Function', 'unit': 'score'},
    'Age': {'type': int, 'min': 10, 'max': 120, 'label': 'Age', 'unit': 'years'}
}

FEATURE_COLUMNS = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

def validate_and_parse_inputs(form_data):
    """
    Validates form inputs against physiological boundaries and data types.
    Returns: (parsed_dict, error_list)
    """
    errors = []
    parsed_values = {}

    for feature, specs in FEATURE_SPECS.items():
        val_raw = form_data.get(feature, None)
        if val_raw is None or (isinstance(val_raw, str) and val_raw.strip() == ''):
            errors.append(f"Field '{specs['label']}' is required.")
            continue

        try:
            if specs['type'] is int:
                val = int(val_raw)
            else:
                val = float(val_raw)
        except (ValueError, TypeError):
            errors.append(f"'{specs['label']}' must be a valid number.")
            continue

        if val < specs['min'] or val > specs['max']:
            errors.append(
                f"'{specs['label']}' value ({val}) is out of reasonable range "
                f"[{specs['min']} - {specs['max']} {specs['unit']}]."
            )
            continue

        parsed_values[feature] = val

    return parsed_values, errors

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        metadata=metadata,
        feature_specs=FEATURE_SPECS,
        form_values={},
        prediction=None,
        errors=[]
    )

@app.route('/predict', methods=['POST'])
def predict():
    form_values = {f: request.form.get(f, '') for f in FEATURE_COLUMNS}
    parsed_values, errors = validate_and_parse_inputs(request.form)

    if errors:
        return render_template(
            'index.html',
            metadata=metadata,
            feature_specs=FEATURE_SPECS,
            form_values=form_values,
            prediction=None,
            errors=errors
        )

    # Build single-sample DataFrame with exact feature columns
    input_df = pd.DataFrame([parsed_values], columns=FEATURE_COLUMNS)

    # Perform prediction using the saved pipeline
    pred_class_idx = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    positive_prob = float(probabilities[1]) * 100
    negative_prob = float(probabilities[0]) * 100

    predicted_label = "Diabetes-positive class" if pred_class_idx == 1 else "Diabetes-negative class"

    prediction_result = {
        'class_index': pred_class_idx,
        'label': predicted_label,
        'positive_probability': round(positive_prob, 2),
        'negative_probability': round(negative_prob, 2),
        'primary_probability': round(positive_prob if pred_class_idx == 1 else negative_prob, 2),
        'input_summary': parsed_values
    }

    return render_template(
        'index.html',
        metadata=metadata,
        feature_specs=FEATURE_SPECS,
        form_values=form_values,
        prediction=prediction_result,
        errors=[]
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic inference verification."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON body'}), 400

    parsed_values, errors = validate_and_parse_inputs(data)
    if errors:
        return jsonify({'status': 'error', 'errors': errors}), 422

    input_df = pd.DataFrame([parsed_values], columns=FEATURE_COLUMNS)
    pred_idx = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    return jsonify({
        'status': 'success',
        'prediction': 'Diabetes-positive class' if pred_idx == 1 else 'Diabetes-negative class',
        'class_index': pred_idx,
        'estimated_probability': round(float(probabilities[1]) * 100, 2),
        'probabilities': {
            'negative': round(float(probabilities[0]) * 100, 2),
            'positive': round(float(probabilities[1]) * 100, 2)
        }
    })

@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    """Return model evaluation and dataset metrics in JSON format."""
    return jsonify(metadata)

if __name__ == '__main__':
    # Local only execution without cloud services
    print("Starting Diabetes Prediction System local server...")
    print("Access via web browser at: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
