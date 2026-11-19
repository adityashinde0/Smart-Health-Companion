"""
Machine Learning Pipeline for Cardiovascular Disease Risk Prediction.
Trains a calibrated classifier on clinical indicators from the Cleveland Heart Disease dataset.
"""

from typing import Dict, Any, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar > 120 mg/dL",
    "restecg": "Resting Electrocardiogram (0-2)",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise-Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression (Exercise vs Rest)",
    "slope": "Slope of Peak Exercise ST (0-2)",
    "ca": "Major Vessels Colored by Fluoroscopy (0-4)",
    "thal": "Thalassemia Type (0-3)"
}


def load_dataset(csv_path_or_buffer: Any) -> pd.DataFrame:
    """Load and validate cardiovascular dataset."""
    df = pd.read_csv(csv_path_or_buffer)
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded dataset is missing required clinical columns: {missing_cols}")
    if "target" not in df.columns:
        raise ValueError("Dataset must contain the 'target' column indicating heart disease presence.")
    return df


def train_heart_model(
    df: pd.DataFrame,
    model_type: str = "rf",
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Train an ML classifier on the dataset.
    Returns:
        - Trained model/pipeline
        - Evaluation metrics dictionary
        - Feature importance DataFrame
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    if model_type == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ])
        model.fit(X_train, y_train)
        coefs = np.abs(model.named_steps["classifier"].coef_[0])
        importances = coefs / np.sum(coefs)
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        importances = model.feature_importances_

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "total_samples": len(df),
        "test_samples": len(y_test)
    }

    feature_imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[col] for col in FEATURE_COLS],
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return model, metrics, feature_imp_df


def predict_patient_risk(
    model: Any,
    patient_data: Dict[str, Any]
) -> Tuple[float, int, pd.DataFrame]:
    """
    Predict risk probability (%) for a single patient dictionary.
    Returns:
        - risk_percentage (float 0.0 to 100.0)
        - binary_prediction (0: Low Risk / No Disease, 1: High Risk / Disease Present)
        - patient_feature_df (DataFrame matching input order)
    """
    row = {col: [patient_data[col]] for col in FEATURE_COLS}
    input_df = pd.DataFrame(row)
    proba = float(model.predict_proba(input_df)[0][1]) * 100.0
    pred = int(model.predict(input_df)[0])
    return round(proba, 1), pred, input_df


def save_model(model: Any, metrics: Dict[str, Any], filepath: str = "heart_model.pkl") -> None:
    """Persist the trained model and metrics artifact."""
    payload = {
        "model": model,
        "metrics": metrics,
        "features": FEATURE_COLS
    }
    joblib.dump(payload, filepath)
    print(f"Model artifact successfully saved to {filepath}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "heart.csv")
    print(f"Loading cardiovascular dataset from {data_path}...")
    dataset = load_dataset(data_path)
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns.")

    print("\nTraining Random Forest Classifier...")
    rf_model, rf_metrics, rf_feat_imp = train_heart_model(dataset, model_type="rf")
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, ROC AUC: {rf_metrics['roc_auc']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    print("\nTop 5 Clinical Risk Factors by Importance:")
    for idx, row in rf_feat_imp.head(5).iterrows():
        print(f"  {idx+1}. {row['label']}: {row['importance']*100:.2f}%")

    model_output_path = os.path.join(current_dir, "heart_model.pkl")
    save_model(rf_model, rf_metrics, model_output_path)
