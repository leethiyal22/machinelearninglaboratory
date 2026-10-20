"""
build_report_pdf.py - Generate the complete 11-page project report PDF matching the sample report model.
Output: Diabetes_Prediction_System_Report.pdf
"""

import os
import subprocess
import tempfile
import pypdf

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_DIR = os.path.abspath('.').replace('\\', '/')
OUTPUT_PDF = os.path.abspath('Diabetes_Prediction_System_Report.pdf')

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Diabetes Prediction System Using Machine Learning - Project Report</title>
    <style>
        @page {
            size: A4 portrait;
            margin: 14mm 16mm 14mm 16mm;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: "Times New Roman", Times, serif;
            font-size: 10.5pt;
            line-height: 1.34;
            color: #000000;
            margin: 0;
            padding: 0;
            text-align: justify;
        }

        .page {
            page-break-after: always;
            height: 268mm;
            max-height: 268mm;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .page:last-child {
            page-break-after: avoid;
        }

        h1.title {
            font-size: 13.5pt;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 3px;
            line-height: 1.25;
        }

        h2.subtitle {
            font-size: 10.5pt;
            font-style: italic;
            font-weight: normal;
            text-align: center;
            margin-top: 0;
            margin-bottom: 16px;
        }

        h3.section-header {
            font-size: 11pt;
            font-weight: bold;
            margin-top: 12px;
            margin-bottom: 6px;
            text-align: left;
        }

        p {
            margin-top: 0;
            margin-bottom: 9px;
        }

        ol, ul {
            margin-top: 0;
            margin-bottom: 10px;
            padding-left: 22px;
        }

        li {
            margin-bottom: 4px;
        }

        .code-box {
            background-color: #fafafa;
            border: 1px solid #d0d7de;
            border-radius: 3px;
            padding: 8px 10px;
            font-family: Consolas, "Courier New", monospace;
            font-size: 8pt;
            line-height: 1.3;
            color: #1e293b;
            margin-bottom: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .code-title {
            font-size: 10pt;
            font-weight: bold;
            margin-top: 8px;
            margin-bottom: 3px;
            color: #0f172a;
        }

        .figure-container {
            text-align: center;
            margin-top: 8px;
            margin-bottom: 12px;
        }

        .figure-img {
            max-width: 95%;
            height: auto;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: inline-block;
        }

        .figure-caption {
            font-size: 9.5pt;
            font-style: italic;
            text-align: center;
            margin-top: 5px;
            color: #1e293b;
        }
    </style>
</head>
<body>

    <!-- ==================== PAGE 1 ==================== -->
    <div class="page">
        <h1 class="title">Diabetes Prediction from Physiological Health Data using a Multi-Model Classifier System</h1>
        <h2 class="subtitle">(Diabetes Risk Classification using Machine Learning)</h2>

        <h3 class="section-header">Problem Statement:</h3>
        <p>
            Diabetes mellitus is one of the leading global causes of chronic metabolic impairment and long-term physiological deterioration. Often termed a "silent epidemic," diabetes frequently develops asymptomatically over several years, during which progressive cellular damage to pancreatic beta cells and microvascular structures takes place. Manual clinical screening through oral glucose tolerance tests and fasting blood draws is accurate but resource-intensive, time-consuming, and difficult to deploy across large asymptomatic populations, particularly in resource-limited rural communities.
        </p>
        <p>
            There is, therefore, a pressing need for an automated, fast, accessible, and transparent system that can analyze routine physiological measurements (such as glucose, blood pressure, BMI, insulin, and age) and estimate whether an individual falls into a diabetes-positive or diabetes-negative risk category with a measurable confidence score, so that suspected high-risk individuals can be flagged early for formal clinical evaluation. This project designs and implements a web-based Diabetes Prediction System that executes domain-specific data preprocessing, invalid-zero imputation, exploratory data analysis, and classification using supervised machine learning algorithms, and evaluates the trained models using standard classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix).
        </p>

        <h3 class="section-header">Objectives:</h3>
        <ol>
            <li>To collect and preprocess a labelled physiological dataset of diabetes indicators from the benchmark Pima Indians Diabetes Database.</li>
            <li>To detect and treat biologically invalid zero measurements in physiological features (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) while preserving valid zeros (Pregnancies).</li>
            <li>To perform Exploratory Data Analysis (EDA) and visualize outcome balance, risk factors, and Pearson correlation coefficients.</li>
            <li>To design, train, and save a supervised classification pipeline using scikit-learn on an identical stratified 80:20 train-test split.</li>
            <li>To build a Flask-based backend and modern web frontend that allows a user to input diagnostic parameters and view the estimated category with probability scores.</li>
            <li>To evaluate the trained model on an unseen held-out test set and report Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix.</li>
        </ol>

        <h3 class="section-header">Software / Hardware Required:</h3>
        <ul>
            <li>• Operating System: Windows / Linux / macOS</li>
            <li>• IDE: Visual Studio Code</li>
            <li>• Language: Python 3.x</li>
            <li>• Backend Framework: Flask (REST API)</li>
        </ul>
    </div>

    <!-- ==================== PAGE 2 ==================== -->
    <div class="page">
        <ul>
            <li>• Machine Learning: scikit-learn (RandomForestClassifier, DecisionTreeClassifier, LogisticRegression)</li>
            <li>• Data Processing & Visualization: Pandas, NumPy, Matplotlib, Seaborn</li>
            <li>• Pipeline Persistence: Joblib</li>
            <li>• Frontend: HTML5, CSS3, JavaScript</li>
            <li>• Dataset: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases)</li>
            <li>• Version Control: Git</li>
        </ul>

        <h3 class="section-header" style="margin-top: 15px;">Project Directory Structure:</h3>
        <p>
            The project (“DiabetesPredictionSystem”) is organized as a Flask backend with a separate dataset, trained-model pipeline, static visualizations, and templates folder, as shown below.
        </p>

        <div class="figure-container" style="margin-top: 18px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig1_vscode_tree.png" alt="Project Directory Structure" style="max-height: 440px; width: auto; border: 1px solid #333333; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
            <div class="figure-caption" style="margin-top: 10px;">Fig 1: Project directory structure of DiabetesPredictionSystem in Visual Studio Code</div>
        </div>
    </div>

    <!-- ==================== PAGE 3 ==================== -->
    <div class="page">
        <h3 class="section-header">Procedure:</h3>
        <ol>
            <li>Collect the labelled Pima Indians Diabetes dataset (768 patient records with 8 diagnostic features) and organize it into <code>data/diabetes.csv</code>.</li>
            <li>Execute <code>train_model.py</code> to inspect dataset records, verify data types, and check for missing values or duplicates.</li>
            <li>Identify biological zero anomalies (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) and treat them as missing measurements while preserving valid zeros for Pregnancies.</li>
            <li>Construct a leakage-free preprocessing pipeline using <code>ColumnTransformer</code> with <code>SimpleImputer(strategy='median')</code> and <code>StandardScaler</code>.</li>
            <li>Perform a stratified 80:20 train-test split (614 training rows, 154 test rows) using fixed random seed (42) to preserve exact class proportion.</li>
            <li>Generate exploratory data analysis visualizations (Outcome distribution, Glucose vs Outcome, BMI distribution, Age distribution, Pearson correlation heatmap) and save to <code>static/plots/</code>.</li>
            <li>Train candidate supervised classifiers: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier on the identical training split.</li>
            <li>Evaluate each model on the held-out test split, computing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix.</li>
            <li>Select the optimal model based on ROC-AUC (0.8213) and F1-score (0.6726) to guard against false negatives, and save the pipeline to <code>models/diabetes_model.pkl</code> via Joblib.</li>
            <li>Launch the Flask application (<code>app.py</code>), input diagnostic indicators into the web form, validate ranges, and display predictions and estimated probabilities.</li>
        </ol>

        <h3 class="section-header">Implementation Code:</h3>
        <p style="font-size: 9.5pt; font-style: italic; color: #334155; margin-bottom: 6px;">
            Representative implementation of the key modules used in the project, based on the pipeline shown in the application (adapt variable/file names to match your exact source files).
        </p>

        <div class="code-title">1. Data Preprocessing & Pipeline Construction (train_model.py)</div>
        <div class="code-box">import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset and separate features and target
df = pd.read_csv('data/diabetes.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Define physiological features with invalid zeros vs natural valid zeros
zero_invalid_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
normal_features = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age']

# ColumnTransformer: treats 0 as missing in biological features, preserves pregnancies
preprocessor = ColumnTransformer(
    transformers=[
        ('zero_imputer', Pipeline([
            ('imputer', SimpleImputer(missing_values=0, strategy='median')),
            ('scaler', StandardScaler())
        ]), zero_invalid_features),
        ('normal_imputer', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), normal_features)
    ],
    remainder='drop'
)</div>
    </div>

    <!-- ==================== PAGE 4 ==================== -->
    <div class="page">
        <div class="code-title">2. Exploratory Data Analysis & Visualization (train_model.py)</div>
        <div class="code-box">import matplotlib.pyplot as plt
import seaborn as sns, os

def generate_eda_visualizations(df):
    os.makedirs('static/plots', exist_ok=True)
    
    # Outcome Class Distribution
    plt.figure(figsize=(7, 5), dpi=150)
    counts = df['Outcome'].value_counts()
    plt.bar(['Negative (0)', 'Positive (1)'], counts.values, color=['#10B981', '#EF4444'])
    plt.title('Diabetes Outcome Class Distribution')
    plt.savefig('static/plots/outcome_distribution.png', bbox_inches='tight')
    plt.close()

    # Glucose Level vs Outcome (filtering invalid 0s)
    valid_glucose = df[df['Glucose'] > 0]
    plt.figure(figsize=(7.5, 5.2), dpi=150)
    sns.boxplot(x='Outcome', y='Glucose', data=valid_glucose, palette=['#A7F3D0', '#FECDD3'])
    plt.title('Glucose Level by Diabetes Outcome')
    plt.savefig('static/plots/glucose_vs_outcome.png', bbox_inches='tight')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(8.5, 6.8), dpi=150)
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues')
    plt.title('Feature Correlation Matrix Heatmap')
    plt.savefig('static/plots/correlation_heatmap.png', bbox_inches='tight')
    plt.close()</div>

        <div class="code-title">3. Model Training & Multi-Model Evaluation (train_model.py)</div>
        <div class="code-box">from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Stratified 80:20 Train-Test Split with fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

candidate_models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)
}

trained_pipelines = {}
for name, clf in candidate_models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Evaluate on held-out test data
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print(f"{name} -> Acc: {accuracy_score(y_test, y_pred):.4f}, AUC: {roc_auc_score(y_test, y_prob):.4f}")</div>
    </div>

    <!-- ==================== PAGE 5 ==================== -->
    <div class="page">
        <div class="code-title">4. Model Selection & Persistence (train_model.py)</div>
        <div class="code-box"># Select best model based on ROC-AUC with balanced sensitivity
best_model_name = 'Random Forest'
best_pipeline = trained_pipelines[best_model_name]

# Save pipeline (including fitted preprocessing) via Joblib
joblib.dump(best_pipeline, 'models/diabetes_model.pkl')
print(f"Saved selected best pipeline ({best_model_name}) to models/diabetes_model.pkl")</div>

        <div class="code-title">5. Prediction Route & Web Service (app.py)</div>
        <div class="code-box">from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load('models/diabetes_model.pkl')

FEATURE_COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    form_values = {f: request.form.get(f, '') for f in FEATURE_COLUMNS}
    parsed_values, errors = validate_and_parse_inputs(request.form)

    if errors:
        return render_template('index.html', errors=errors, form_values=form_values)

    # Convert parsed inputs to single-row DataFrame
    input_df = pd.DataFrame([parsed_values], columns=FEATURE_COLUMNS)

    # Generate prediction and probabilities from pipeline
    pred_class_idx = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    positive_prob = round(float(probabilities[1]) * 100, 2)
    negative_prob = round(float(probabilities[0]) * 100, 2)

    predicted_label = "Diabetes-positive class" if pred_class_idx == 1 else "Diabetes-negative class"

    prediction_result = {
        'class_index': pred_class_idx,
        'label': predicted_label,
        'positive_probability': positive_prob,
        'negative_probability': negative_prob,
        'input_summary': parsed_values
    }

    return render_template('index.html', prediction=prediction_result, form_values=form_values)</div>
    </div>

    <!-- ==================== PAGE 6 ==================== -->
    <div class="page">
        <div class="code-title">6. Input Validation & Physiological Range Verification (app.py)</div>
        <div class="code-box">FEATURE_SPECS = {
    'Pregnancies': {'type': int, 'min': 0, 'max': 25, 'label': 'Pregnancies'},
    'Glucose': {'type': float, 'min': 0.0, 'max': 400.0, 'label': 'Glucose Level'},
    'BloodPressure': {'type': float, 'min': 0.0, 'max': 240.0, 'label': 'Blood Pressure'},
    'SkinThickness': {'type': float, 'min': 0.0, 'max': 110.0, 'label': 'Skin Thickness'},
    'Insulin': {'type': float, 'min': 0.0, 'max': 900.0, 'label': 'Insulin Level'},
    'BMI': {'type': float, 'min': 0.0, 'max': 80.0, 'label': 'BMI'},
    'DiabetesPedigreeFunction': {'type': float, 'min': 0.01, 'max': 3.5, 'label': 'Diabetes Pedigree'},
    'Age': {'type': int, 'min': 10, 'max': 120, 'label': 'Age'}
}

def validate_and_parse_inputs(form_data):
    errors, parsed_values = [], {}
    for feature, specs in FEATURE_SPECS.items():
        val_raw = form_data.get(feature, None)
        if val_raw is None or (isinstance(val_raw, str) and val_raw.strip() == ''):
            errors.append(f"Field '{specs['label']}' is required.")
            continue
        try:
            val = specs['type'](val_raw)
        except (ValueError, TypeError):
            errors.append(f"'{specs['label']}' must be a valid number.")
            continue
        if val < specs['min'] or val > specs['max']:
            errors.append(f"'{specs['label']}' value ({val}) is out of physiological range.")
            continue
        parsed_values[feature] = val
    return parsed_values, errors</div>

        <h3 class="section-header" style="margin-top: 14px;">Output (Step-by-Step Screenshots):</h3>

        <div class="figure-container" style="margin-top: 8px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig2_dataset_overview.png" alt="Fig 2: Dataset Overview" class="figure-img" style="max-height: 220px;">
            <div class="figure-caption">Fig 2: Step 1 — Load Input Data & Dataset Overview: recorded observations, 8 numerical features, and first 5 patient records</div>
        </div>
    </div>

    <!-- ==================== PAGE 7 ==================== -->
    <div class="page">
        <div class="figure-container">
            <img src="file:///{BASE_DIR}/static/screenshots/fig3_data_preprocessing.png" alt="Fig 3: Data Preprocessing" class="figure-img" style="max-height: 235px;">
            <div class="figure-caption">Fig 3: Step 2 — Data Preprocessing: missing/invalid zero analysis and stratified train-test split details</div>
        </div>

        <div class="figure-container" style="margin-top: 14px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig4_eda_distributions.png" alt="Fig 4: EDA Distributions" class="figure-img" style="max-height: 225px;">
            <div class="figure-caption">Fig 4: Step 3 — Exploratory Data Analysis: Outcome distribution, Glucose vs Outcome, and BMI distribution</div>
        </div>
    </div>

    <!-- ==================== PAGE 8 ==================== -->
    <div class="page">
        <div class="figure-container">
            <img src="file:///{BASE_DIR}/static/screenshots/fig5_eda_heatmap.png" alt="Fig 5: EDA Heatmap & Age" class="figure-img" style="max-height: 230px;">
            <div class="figure-caption">Fig 5: Step 3 — Exploratory Data Analysis: Age distribution and feature correlation matrix heatmap</div>
        </div>

        <div class="figure-container" style="margin-top: 14px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig6_model_training.png" alt="Fig 6: Model Training & Comparison" class="figure-img" style="max-height: 255px;">
            <div class="figure-caption">Fig 6: Step 4 — Model Training & Comparison: multi-algorithm evaluation table and comparative bar chart</div>
        </div>
    </div>

    <!-- ==================== PAGE 9 ==================== -->
    <div class="page">
        <div class="figure-container">
            <img src="file:///{BASE_DIR}/static/screenshots/fig7_model_evaluation.png" alt="Fig 7: Model Evaluation Metrics" class="figure-img" style="max-height: 250px;">
            <div class="figure-caption">Fig 7: Step 5 — Selected Model Evaluation: held-out test metrics (Accuracy 75.97%, Precision 64.41%, Recall 70.37%, F1 0.6726, ROC-AUC 0.8213), confusion matrix and ROC curves</div>
        </div>

        <div class="figure-container" style="margin-top: 14px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig8_prediction_form.png" alt="Fig 8: Diabetes Prediction Form" class="figure-img" style="max-height: 230px;">
            <div class="figure-caption">Fig 8: Step 6 — Diabetes Prediction Form: input parameter interface with physiological units and sample prefill buttons</div>
        </div>
    </div>

    <!-- ==================== PAGE 10 ==================== -->
    <div class="page">
        <div class="figure-container">
            <img src="file:///{BASE_DIR}/static/screenshots/fig9_prediction_negative.png" alt="Fig 9: Low Risk Prediction Result" class="figure-img" style="max-height: 245px;">
            <div class="figure-caption">Fig 9: Step 6 — Classification Result: Diabetes-negative class (Healthy/Low-Risk), 89.7% negative probability (10.27% risk)</div>
        </div>

        <div class="figure-container" style="margin-top: 14px;">
            <img src="file:///{BASE_DIR}/static/screenshots/fig10_prediction_positive.png" alt="Fig 10: High Risk Prediction Result" class="figure-img" style="max-height: 245px;">
            <div class="figure-caption">Fig 10: Step 6 — Classification Result: Diabetes-positive class (Elevated Risk), 80.1% positive probability</div>
        </div>
    </div>

    <!-- ==================== PAGE 11 ==================== -->
    <div class="page">
        <div class="figure-container">
            <img src="file:///{BASE_DIR}/static/screenshots/fig11_api_response.png" alt="Fig 11: REST API Verification" class="figure-img" style="max-height: 165px;">
            <div class="figure-caption">Fig 11: Step 7 — Verification of Local REST API (/api/predict) and JSON inference response</div>
        </div>

        <h3 class="section-header" style="margin-top: 18px;">Result:</h3>
        <p>
            A complete, working web-based Diabetes Prediction System was successfully designed and implemented using a Flask backend, a Random Forest classification pipeline (scikit-learn), and an interactive light-themed web interface. Authentic clinical measurements from the Pima Indians Diabetes Database were preprocessed using domain-aware median imputation on physiologically invalid zero measurements (Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI) while preserving valid natural zeros (Pregnancies), followed by standard feature scaling with strict elimination of data leakage.
        </p>
        <p>
            Three classification algorithms (Logistic Regression, Decision Tree Classifier, and Random Forest Classifier) were trained and compared on an identical stratified 80:20 split. The Random Forest model demonstrated superior discriminative capability, achieving an <strong>Accuracy of 75.97%</strong>, <strong>Precision of 64.41%</strong>, <strong>Recall of 70.37%</strong>, an <strong>F1-Score of 0.6726</strong>, and an <strong>ROC-AUC of 0.8213</strong> on the held-out test set of 154 observations. The system validates real-time user inputs, computes predicted classification categories with estimated model probabilities, and provides a programmatic JSON REST API alongside clear educational and non-diagnostic medical disclaimers.
        </p>
    </div>

</body>
</html>"""

def generate_pdf():
    html_content = HTML_TEMPLATE.replace('{BASE_DIR}', BASE_DIR)
    temp_html = os.path.join(tempfile.gettempdir(), 'report_template.html')
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Compiling PDF to: {OUTPUT_PDF}...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=2000",
        f"--print-to-pdf={OUTPUT_PDF}",
        temp_html
    ]

    subprocess.run(cmd, check=True)
    if os.path.exists(OUTPUT_PDF):
        reader = pypdf.PdfReader(OUTPUT_PDF)
        total_pages = len(reader.pages)
        size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
        print(f"SUCCESS: Report PDF generated! Total Pages: {total_pages}, Size: {size_mb:.2f} MB")
        return total_pages
    return 0

if __name__ == '__main__':
    generate_pdf()
