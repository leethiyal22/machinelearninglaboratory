"""
train_model.py - End-to-end Machine Learning Training & Evaluation Pipeline
Diabetes Prediction System Using Machine Learning

Workflow:
1. Problem Definition: Binary classification for diabetes risk indication.
2. Data Collection: Load Pima Indians Diabetes Database from data/diabetes.csv.
3. Data Preprocessing:
   - Detect invalid zero values in biological measurements (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
   - Maintain valid zeros (Pregnancies).
   - ColumnTransformer with median imputation and standard scaling.
4. Exploratory Data Analysis (EDA):
   - Outcome distribution, Glucose vs Outcome, BMI distribution, Age distribution, Correlation heatmap.
5. Train-Test Split: 80:20 stratified split with fixed random seed (42).
6. Model Training & Comparison:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
7. Model Evaluation:
   - Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
   - Confusion Matrix & ROC curves saved to static/plots/.
8. Model Selection & Persistence:
   - Best model selected by ROC-AUC and F1-score.
   - Saved to models/diabetes_model.pkl via Joblib.
   - Metadata exported to models/model_metadata.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Styling configuration for publication-quality, modern light-themed plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI, Helvetica, Arial, sans-serif',
    'font.family': 'sans-serif',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CBD5E1',
    'axes.labelcolor': '#1E293B',
    'text.color': '#1E293B',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'grid.color': '#E2E8F0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

PALETTE_OUTCOME = {0: '#10B981', 1: '#EF4444'}  # 0: Green (Negative), 1: Red (Positive)
PALETTE_MODELS = ['#2563EB', '#0D9488', '#8B5CF6']

DATA_PATH = os.path.join('data', 'diabetes.csv')
MODEL_PATH = os.path.join('models', 'diabetes_model.pkl')
METADATA_PATH = os.path.join('models', 'model_metadata.json')
PLOTS_DIR = os.path.join('static', 'plots')

def ensure_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please ensure data/diabetes.csv is downloaded before running train_model.py."
        )
    df = pd.read_csv(DATA_PATH)
    return df

def generate_eda_visualizations(df):
    """Generate high-resolution, light-themed EDA plots based on real recorded measurements."""
    print("Generating EDA visualizations...")

    # 1. Diabetes Outcome Distribution
    plt.figure(figsize=(7, 5), dpi=150)
    counts = df['Outcome'].value_counts()
    percentages = df['Outcome'].value_counts(normalize=True) * 100
    labels = ['Negative (0)', 'Positive (1)']
    bar_colors = [PALETTE_OUTCOME[0], PALETTE_OUTCOME[1]]

    bars = plt.bar(labels, counts.values, color=bar_colors, edgecolor='#94A3B8', width=0.55, linewidth=1.2)
    plt.title('Diabetes Outcome Class Distribution', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('Class Category', fontsize=12, labelpad=10)
    plt.ylabel('Number of Patients', fontsize=12, labelpad=10)
    plt.ylim(0, max(counts.values) * 1.18)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f'{count} ({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1E293B'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'outcome_distribution.png'), dpi=150)
    plt.close()

    # 2. Glucose Level vs Diabetes Outcome (Boxplot + Stripplot overlay)
    plt.figure(figsize=(7.5, 5.2), dpi=150)
    # Filter out biologically invalid zeros for realistic visualization
    valid_glucose_df = df[df['Glucose'] > 0].copy()
    valid_glucose_df['Outcome_Label'] = valid_glucose_df['Outcome'].map({0: 'Negative (0)', 1: 'Positive (1)'})
    
    sns.boxplot(
        x='Outcome_Label',
        y='Glucose',
        hue='Outcome_Label',
        legend=False,
        data=valid_glucose_df,
        palette=['#A7F3D0', '#FECDD3'],
        width=0.45,
        boxprops=dict(edgecolor='#334155', linewidth=1.4),
        medianprops=dict(color='#0F172A', linewidth=2.0)
    )
    sns.stripplot(
        x='Outcome_Label',
        y='Glucose',
        hue='Outcome_Label',
        legend=False,
        data=valid_glucose_df,
        palette=[PALETTE_OUTCOME[0], PALETTE_OUTCOME[1]],
        alpha=0.35,
        jitter=0.2,
        size=4.5
    )
    plt.title('Glucose Level by Diabetes Outcome', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('Diabetes Outcome', fontsize=12, labelpad=10)
    plt.ylabel('Plasma Glucose Concentration (mg/dL)', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'glucose_vs_outcome.png'), dpi=150)
    plt.close()

    # 3. BMI Distribution (KDE + Histograms by Outcome)
    plt.figure(figsize=(7.5, 5.2), dpi=150)
    valid_bmi_df = df[df['BMI'] > 0]
    sns.histplot(
        data=valid_bmi_df,
        x='BMI',
        hue='Outcome',
        palette=PALETTE_OUTCOME,
        kde=True,
        bins=30,
        element='step',
        alpha=0.4,
        linewidth=1.5
    )
    plt.title('Body Mass Index (BMI) Distribution by Outcome', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('BMI (kg/m²)', fontsize=12, labelpad=10)
    plt.ylabel('Patient Count', fontsize=12, labelpad=10)
    plt.legend(title='Outcome', labels=['Positive (1)', 'Negative (0)'])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bmi_distribution.png'), dpi=150)
    plt.close()

    # 4. Age Distribution by Outcome
    plt.figure(figsize=(7.5, 5.2), dpi=150)
    sns.histplot(
        data=df,
        x='Age',
        hue='Outcome',
        palette=PALETTE_OUTCOME,
        kde=True,
        bins=25,
        element='step',
        alpha=0.45,
        linewidth=1.5
    )
    plt.title('Age Distribution of Patients by Outcome', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('Age (Years)', fontsize=12, labelpad=10)
    plt.ylabel('Patient Count', fontsize=12, labelpad=10)
    plt.legend(title='Outcome', labels=['Positive (1)', 'Negative (0)'])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution.png'), dpi=150)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(8.5, 6.8), dpi=150)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        center=0,
        square=True,
        linewidths=1.0,
        linecolor='#FFFFFF',
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    plt.title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    print("EDA visualizations generated successfully.")

def train_and_evaluate():
    ensure_directories()
    df = load_dataset()

    # Preprocessing inspection
    total_records = len(df)
    duplicates = int(df.duplicated().sum())
    missing_nans = df.isna().sum().to_dict()

    # Features where 0 indicates missing/invalid biological measurement
    zero_invalid_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # Features where 0 is natural/valid
    normal_features = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age']

    zeros_summary = {}
    for col in df.columns:
        zeros_count = int((df[col] == 0).sum())
        zeros_summary[col] = {
            'count': zeros_count,
            'percent': round(float((df[col] == 0).mean() * 100), 2),
            'treated_as_missing': col in zero_invalid_features
        }

    # Generate EDA plots
    generate_eda_visualizations(df)

    # Separation of features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # 80:20 Train-Test split with stratification and fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)} rows | Test set size: {len(X_test)} rows")

    # Pipeline design:
    # 1. Zero-imputation pipeline: replaces 0 with median of non-zero values on training set, then standard scales.
    # 2. Normal pipeline: retains 0, standard scales.
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
    )

    # Model candidates
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    roc_curves = {}
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train strictly on X_train to prevent any data leakage
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions on held-out test split
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr.tolist(), tpr.tolist(), auc)

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm
        }
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Model selection based on ROC-AUC with F1-score tie-breaking:
    # Rationale: In clinical screening, discriminative power across decision thresholds (ROC-AUC)
    # and balance between false negatives and precision (F1-score) are essential.
    best_model_name = max(results.keys(), key=lambda m: (results[m]['roc_auc'], results[m]['f1_score']))
    print(f"\nSelected Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']}, F1: {results[best_model_name]['f1_score']})")

    # Save the best model pipeline
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved selected pipeline to {MODEL_PATH}")

    # Generate Model Comparison Chart
    plt.figure(figsize=(9, 5.5), dpi=150)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics_names))
    width = 0.25

    for idx, (m_name, color) in enumerate(zip(candidate_models.keys(), PALETTE_MODELS)):
        vals = [
            results[m_name]['accuracy'],
            results[m_name]['precision'],
            results[m_name]['recall'],
            results[m_name]['f1_score'],
            results[m_name]['roc_auc']
        ]
        rects = plt.bar(x + (idx - 1) * width, vals, width, label=m_name, color=color, edgecolor='#334155', linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.015, f"{height:.2f}",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B', rotation=90)

    plt.title('Classification Algorithms Performance Comparison', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12, labelpad=10)
    plt.xticks(x, metrics_names, fontsize=11, fontweight='600')
    plt.ylim(0, 1.15)
    plt.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()

    # Generate Confusion Matrix for Best Model
    best_cm = np.array(results[best_model_name]['confusion_matrix'])
    plt.figure(figsize=(6, 5), dpi=150)
    annot_labels = np.array([
        [f"TN\n{best_cm[0,0]}", f"FP\n{best_cm[0,1]}"],
        [f"FN\n{best_cm[1,0]}", f"TP\n{best_cm[1,1]}"]
    ])
    sns.heatmap(
        best_cm,
        annot=annot_labels,
        fmt="",
        cmap='Blues',
        cbar=True,
        linewidths=1.5,
        linecolor='#FFFFFF',
        xticklabels=['Predicted Negative (0)', 'Predicted Positive (1)'],
        yticklabels=['Actual Negative (0)', 'Actual Positive (1)'],
        annot_kws={"fontsize": 13, "fontweight": "bold"}
    )
    plt.title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    plt.ylabel('Actual Class', fontsize=11, labelpad=8)
    plt.xlabel('Predicted Class', fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Generate ROC Curve for all models
    plt.figure(figsize=(7, 5.5), dpi=150)
    for (m_name, color) in zip(candidate_models.keys(), PALETTE_MODELS):
        fpr_vals, tpr_vals, auc_score = roc_curves[m_name]
        is_best = (m_name == best_model_name)
        plt.plot(
            fpr_vals,
            tpr_vals,
            color=color,
            linewidth=2.5 if is_best else 1.8,
            linestyle='-' if is_best else '--',
            label=f"{m_name} (AUC = {auc_score:.3f})" + (" [Selected]" if is_best else "")
        )

    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

    # Prepare preview of dataset for UI display
    preview_records = df.head(5).to_dict(orient='records')
    summary_stats = df.describe().round(2).to_dict()
    outcome_counts = df['Outcome'].value_counts()

    metadata = {
        'dataset_info': {
            'total_records': total_records,
            'total_features': len(X.columns),
            'feature_names': list(X.columns),
            'target_column': 'Outcome',
            'outcome_counts': {
                'negative_0': int(outcome_counts[0]),
                'positive_1': int(outcome_counts[1]),
                'positive_ratio': round(float(outcome_counts[1] / total_records * 100), 2)
            },
            'duplicates_count': duplicates,
            'missing_nans': missing_nans,
            'zeros_summary': zeros_summary,
            'preview_records': preview_records,
            'summary_stats': summary_stats
        },
        'train_test_split': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'split_ratio': '80:20',
            'random_seed': 42,
            'stratified': True
        },
        'model_comparison': results,
        'selected_model': {
            'name': best_model_name,
            'selection_criterion': 'Highest ROC-AUC score on test set with balanced sensitivity',
            'metrics': results[best_model_name]
        }
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata exported to {METADATA_PATH}")
    print("Model training and evaluation successfully finished.")

if __name__ == '__main__':
    train_and_evaluate()
