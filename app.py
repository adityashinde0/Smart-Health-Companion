"""
Smart Health Companion - Intelligent Cardiovascular Risk Assessment System
Powered by Scikit-Learn Random Forest Classifier and DeepSeek-V3 via CloudRift.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

from train_model import (
    FEATURE_COLS,
    FEATURE_LABELS,
    load_dataset,
    train_heart_model,
    predict_patient_risk
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Health Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Security & Secret Management
# -----------------------------------------------------------------------------
def resolve_api_key() -> Optional[str]:
    """
    Resolve API key in order of priority:
    1. st.secrets["OPENAI_API_KEY"] or st.secrets["openai"]["api_key"]
    2. os.environ["OPENAI_API_KEY"]
    3. Session state user input from sidebar (in-memory only, never saved to disk)
    """
    # 1. Streamlit Secrets
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return str(st.secrets["openai"]["api_key"]).strip()
    except Exception:
        pass

    # 2. Environment Variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 3. In-memory session state from sidebar
    session_key = st.session_state.get("sidebar_api_key")
    if session_key and session_key.strip():
        return session_key.strip()

    return None


# -----------------------------------------------------------------------------
# Dataset & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(uploaded_file=None) -> pd.DataFrame:
    """Load the bundled heart.csv or user uploaded dataset."""
    if uploaded_file is not None:
        return load_dataset(uploaded_file)
    default_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    return load_dataset(default_path)


@st.cache_resource(show_spinner=False)
def get_trained_pipeline(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Train and cache the Random Forest classifier."""
    return train_heart_model(df, model_type="rf")


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def create_risk_gauge(risk_score: float) -> go.Figure:
    """Generate a medical-grade risk indicator gauge."""
    if risk_score < 35.0:
        bar_color = "#28a745"
        risk_level = "Low Risk"
    elif risk_score < 65.0:
        bar_color = "#ffc107"
        risk_level = "Moderate Risk"
    else:
        bar_color = "#dc3545"
        risk_level = "Elevated / High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "%", "font": {"size": 42, "color": bar_color}},
        title={"text": f"<b>Cardiovascular Risk Probability</b><br><span style='font-size:16px;color:{bar_color}'>{risk_level}</span>", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 35], "color": "rgba(40, 167, 69, 0.2)"},
                {"range": [35, 65], "color": "rgba(255, 193, 7, 0.2)"},
                {"range": [65, 100], "color": "rgba(220, 53, 69, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": risk_score
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_distribution_comparison(df: pd.DataFrame, patient_data: Dict[str, Any]) -> go.Figure:
    """Plot patient metrics against healthy vs high-risk dataset distributions."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Serum Cholesterol (mg/dL)",
            "Resting Blood Pressure (mmHg)",
            "Max Heart Rate Achieved (bpm)"
        )
    )

    healthy_df = df[df["target"] == 0]
    risk_df = df[df["target"] == 1]

    metrics_meta = [
        ("chol", "Cholesterol", patient_data.get("chol", 200), 1),
        ("trestbps", "Resting BP", patient_data.get("trestbps", 120), 2),
        ("thalach", "Max HR", patient_data.get("thalach", 150), 3)
    ]

    for col_name, label, patient_val, col_idx in metrics_meta:
        # Healthy cohort distribution
        fig.add_trace(
            go.Box(
                y=healthy_df[col_name],
                name="Healthy (Target=0)",
                marker_color="#28a745",
                boxmean=True,
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )
        # At-Risk cohort distribution
        fig.add_trace(
            go.Box(
                y=risk_df[col_name],
                name="At-Risk (Target=1)",
                marker_color="#dc3545",
                boxmean=True,
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )
        # Patient's position
        fig.add_hline(
            y=patient_val,
            line_dash="dash",
            line_color="#0066cc",
            line_width=2.5,
            annotation_text=f"You: {patient_val}",
            annotation_position="top right",
            annotation_font=dict(color="#0066cc", size=11, family="Arial Black"),
            row=1, col=col_idx
        )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_feature_importance_chart(feat_imp_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top clinical predictors."""
    top_feats = feat_imp_df.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top_feats["importance"] * 100,
        y=top_feats["label"],
        orientation="h",
        marker=dict(
            color=top_feats["importance"],
            colorscale="Viridis",
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in top_feats["importance"] * 100],
        textposition="outside"
    ))
    fig.update_layout(
        title="Top 8 Clinical Feature Contributions to Risk Prediction",
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
        height=320,
        margin=dict(l=10, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# -----------------------------------------------------------------------------
# DeepSeek-V3 LLM Clinical Synthesis
# -----------------------------------------------------------------------------
def generate_ai_clinical_summary(
    api_key: str,
    patient_data: Dict[str, Any],
    risk_score: float,
    pred_label: str,
    top_features: pd.DataFrame
) -> str:
    """Generate professional clinical reasoning report via DeepSeek-V3 on CloudRift."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://inference.cloudrift.ai/v1"
    )

    top_feat_str = ", ".join([
        f"{row['label']} ({row['importance']*100:.1f}% model weight)"
        for _, row in top_features.head(4).iterrows()
    ])

    system_prompt = (
        "You are an expert clinical cardiologist and AI health companion. "
        "Analyze patient biometric data and machine learning predictions with clinical precision, "
        "empathy, and clarity. Format your response into 4 distinct markdown sections: "
        "1. Clinical Assessment & Risk Explanation\n"
        "2. Key Cardiovascular Risk Factors Identified\n"
        "3. Evidence-Based Dietary & Lifestyle Suggestions\n"
        "4. Questions to Ask Your Doctor or Cardiologist\n"
        "Always maintain an objective, supportive medical tone. Do not make definitive diagnostic claims."
    )

    user_prompt = f"""
Patient Profile:
- Age: {patient_data['age']} years
- Biological Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}
- Chest Pain Type (0-3): {patient_data['cp']} (0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal, 3=Asymptomatic)
- Resting Blood Pressure: {patient_data['trestbps']} mmHg
- Serum Cholesterol: {patient_data['chol']} mg/dL
- Fasting Blood Sugar > 120 mg/dL: {'Yes' if patient_data['fbs'] == 1 else 'No'}
- Resting ECG: {patient_data['restecg']}
- Maximum Heart Rate Achieved: {patient_data['thalach']} bpm
- Exercise-Induced Angina: {'Yes' if patient_data['exang'] == 1 else 'No'}
- ST Depression (oldpeak): {patient_data['oldpeak']}
- Slope of ST Segment: {patient_data['slope']}
- Number of Major Vessels (ca): {patient_data['ca']}
- Thalassemia: {patient_data['thal']}
- Reported Symptoms / Notes: {patient_data.get('symptoms', 'None provided')}

Machine Learning Predictive Output:
- Model: Random Forest Classifier (trained on Cleveland Heart Disease Cohort)
- Predicted Risk Score: {risk_score}%
- Classification Status: {pred_label}
- Primary Model Drivers: {top_feat_str}

Please generate an in-depth clinical analysis report based on these parameters.
"""

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content or "No response received."


# -----------------------------------------------------------------------------
# Application Layout & Flow
# -----------------------------------------------------------------------------
def main():
    # Top Medical Disclaimer Callout
    st.warning(
        "⚠️ **Medical Disclaimer:** This application is for educational and research purposes only. "
        "It does not provide medical diagnoses, clinical prescriptions, or treatment advice. "
        "Always consult a board-certified physician or emergency healthcare provider regarding any health condition."
    )

    st.title("🩺 Smart Health Companion")
    st.markdown(
        "**AI-Powered Cardiovascular Risk Prediction & Clinical Insights Platform**  \n"
        "Integrating Scikit-Learn Machine Learning with DeepSeek-V3 LLM for personalized health analytics."
    )

    # -------------------------------------------------------------------------
    # Sidebar: Data Management & Secrets Configuration
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset Selection / Custom Upload
        st.subheader("📁 Data Source")
        uploaded_file = st.file_uploader(
            "Upload Custom Cohort CSV (optional)",
            type=["csv"],
            help="Upload a CSV with the Cleveland clinical schema or use the bundled heart.csv."
        )

        try:
            df = get_dataset(uploaded_file)
            dataset_source = "Uploaded CSV" if uploaded_file else "Bundled Cleveland heart.csv"
            st.success(f"Loaded: {dataset_source} ({len(df)} records)")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

        # Secure API Key Handling
        st.subheader("🔑 LLM Credentials")
        active_api_key = resolve_api_key()

        if active_api_key:
            st.info("API Key active (from secrets / environment / session).")
            with st.expander("Update / Override API Key"):
                new_key = st.text_input(
                    "CloudRift / OpenAI API Key",
                    value="",
                    type="password",
                    placeholder="Enter new key...",
                    key="sidebar_api_key_input"
                )
                if new_key:
                    st.session_state["sidebar_api_key"] = new_key
                    active_api_key = new_key
        else:
            with st.expander("Enter CloudRift / DeepSeek API Key", expanded=True):
                st.markdown(
                    "Enter your API key below for the current session. "
                    "Keys are kept purely in-memory and **never** persisted to disk."
                )
                input_key = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="rift_...",
                    key="manual_api_key"
                )
                if input_key:
                    st.session_state["sidebar_api_key"] = input_key
                    active_api_key = input_key
                    st.success("Session key set!")

        # Train / Fetch Cached ML Model
        try:
            model, model_metrics, feat_imp_df = get_trained_pipeline(df)
        except Exception as e:
            st.error(f"Failed to train ML pipeline: {e}")
            st.stop()

        with st.expander("📊 Model Validation Metrics", expanded=False):
            st.write(f"**Algorithm:** Random Forest Classifier")
            st.write(f"**Test Accuracy:** `{model_metrics['accuracy'] * 100:.2f}%`")
            st.write(f"**ROC-AUC Score:** `{model_metrics['roc_auc']:.4f}`")
            st.write(f"**F1-Score:** `{model_metrics['f1']:.4f}`")
            st.write(f"**Cohort Size:** {model_metrics['total_samples']} patients")

    # -------------------------------------------------------------------------
    # Main Form: Patient Biometrics & Inputs
    # -------------------------------------------------------------------------
    st.subheader("📋 Patient Biometric Profile")
    st.markdown("Provide patient clinical parameters below to compute risk probability and AI insights.")

    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=18, max_value=110, value=52, step=1)
            sex_option = st.selectbox("Biological Sex", ["Male", "Female"])
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130, step=1)
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=1)

        with col2:
            st.markdown("##### 🫀 Cardiac Symptoms")
            cp_option = st.selectbox(
                "Chest Pain Type (cp)",
                [
                    "0: Typical Angina",
                    "1: Atypical Angina",
                    "2: Non-Anginal Pain",
                    "3: Asymptomatic"
                ],
                index=0
            )
            fbs_option = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No (<= 120 mg/dL)", "Yes (> 120 mg/dL)"])
            restecg_option = st.selectbox(
                "Resting ECG (restecg)",
                [
                    "0: Normal",
                    "1: ST-T Wave Abnormality",
                    "2: Left Ventricular Hypertrophy"
                ],
                index=1
            )
            thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150, step=1)

        with col3:
            st.markdown("##### 🔬 Stress Test & Fluoroscopy")
            exang_option = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
            slope_option = st.selectbox(
                "Slope of Peak Exercise ST",
                [
                    "0: Upsloping",
                    "1: Flat",
                    "2: Downsloping"
                ],
                index=1
            )
            ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4], index=0)
            thal_option = st.selectbox(
                "Thalassemia (thal)",
                [
                    "1: Normal",
                    "2: Fixed Defect",
                    "3: Reversible Defect",
                    "0: Other / Unspecified"
                ],
                index=1
            )

        symptoms = st.text_area(
            "Current Symptoms & Clinical Observations (optional)",
            placeholder="e.g. Occasional shortness of breath during brisk walking, family history of hypertension..."
        )

        submit_btn = st.form_submit_button("🔍 Compute Health Risk & Generate AI Analysis", use_container_width=True)

    # Convert Inputs to Normalized Feature Dictionary
    sex = 1 if sex_option == "Male" else 0
    cp = int(cp_option.split(":")[0])
    fbs = 1 if "Yes" in fbs_option else 0
    restecg = int(restecg_option.split(":")[0])
    exang = 1 if exang_option == "Yes" else 0
    slope = int(slope_option.split(":")[0])
    thal = int(thal_option.split(":")[0])

    patient_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "symptoms": symptoms
    }

    # Store patient data in session for download report
    st.session_state["latest_patient_data"] = patient_dict

    # -------------------------------------------------------------------------
    # Analysis & Results Presentation
    # -------------------------------------------------------------------------
    if submit_btn:
        st.markdown("---")
        st.header("📊 Clinical Risk Assessment Results")

        # 1. Run Machine Learning Inference
        risk_score, pred_class, _ = predict_patient_risk(model, patient_dict)
        pred_label = "Elevated Cardiovascular Risk Detected" if pred_class == 1 else "Lower / Normal Cardiovascular Risk"

        st.session_state["latest_risk_score"] = risk_score
        st.session_state["latest_pred_label"] = pred_label

        # 2. Risk Indicators Display
        gauge_col, metrics_col = st.columns([1.2, 1])

        with gauge_col:
            st.plotly_chart(create_risk_gauge(risk_score), use_container_width=True)

        with metrics_col:
            st.markdown("#### Diagnostic Summary")
            if risk_score >= 65.0:
                st.error(f"**High Risk ({risk_score}%)**: Clinical features strongly align with cardiovascular disease profiles.")
            elif risk_score >= 35.0:
                st.warning(f"**Moderate Risk ({risk_score}%)**: Elevated clinical indicators detected. Preventative review advised.")
            else:
                st.success(f"**Low Risk ({risk_score}%)**: Clinical parameters remain largely within low-risk benchmark ranges.")

            st.write(f"- **Algorithm Prediction:** {pred_label}")
            st.write(f"- **Key Contributing Factors:** {', '.join(feat_imp_df.head(3)['label'].tolist())}")
            st.write(f"- **Resting BP Status:** {'Elevated (>120)' if trestbps > 120 else 'Optimal (<=120)'}")
            st.write(f"- **Cholesterol Status:** {'High (>200)' if chol > 200 else 'Desirable (<=200)'}")

        # 3. Interactive Visualizations
        st.subheader("📈 Cohort Benchmark Comparison")
        st.markdown("Your metrics (blue dashed line) benchmarked against healthy vs. high-risk individuals in the clinical dataset:")
        st.plotly_chart(create_distribution_comparison(df, patient_dict), use_container_width=True)

        st.plotly_chart(create_feature_importance_chart(feat_imp_df), use_container_width=True)

        # 4. LLM Synthesis with DeepSeek-V3
        st.markdown("---")
        st.subheader("🤖 DeepSeek-V3 Clinical AI Synthesis")

        resolved_key = resolve_api_key()
        if not resolved_key:
            st.warning(
                "🔑 **LLM Insights Paused:** DeepSeek-V3 clinical summary requires an API key. "
                "Please enter your CloudRift / OpenAI key in the left sidebar to unlock the AI narrative report."
            )
            st.session_state["latest_ai_summary"] = "AI summary not generated: No API key provided."
        else:
            with st.spinner("Synthesizing clinical insights via DeepSeek-V3 on CloudRift..."):
                try:
                    ai_summary = generate_ai_clinical_summary(
                        api_key=resolved_key,
                        patient_data=patient_dict,
                        risk_score=risk_score,
                        pred_label=pred_label,
                        top_features=feat_imp_df
                    )
                    st.session_state["latest_ai_summary"] = ai_summary
                    st.markdown(ai_summary)
                except Exception as ex:
                    st.error(f"DeepSeek-V3 API call failed: {ex}")
                    st.session_state["latest_ai_summary"] = f"API Error: {ex}"

    # -------------------------------------------------------------------------
    # Report Export Button
    # -------------------------------------------------------------------------
    if "latest_risk_score" in st.session_state:
        st.markdown("---")
        st.subheader("📑 Export Patient Report")

        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_info = st.session_state.get("latest_patient_data", patient_dict)
        cur_risk = st.session_state.get("latest_risk_score", "N/A")
        cur_label = st.session_state.get("latest_pred_label", "N/A")
        cur_ai = st.session_state.get("latest_ai_summary", "Not generated")

        report_content = f"""# Smart Health Companion - Patient Clinical Risk Summary Report
**Generated On:** {report_timestamp}

---

## 1. Patient Biometric Inputs
- **Age:** {patient_info.get('age')} years
- **Sex:** {'Male' if patient_info.get('sex') == 1 else 'Female'}
- **Chest Pain Type:** {patient_info.get('cp')}
- **Resting Blood Pressure:** {patient_info.get('trestbps')} mmHg
- **Serum Cholesterol:** {patient_info.get('chol')} mg/dL
- **Fasting Blood Sugar > 120 mg/dL:** {'Yes' if patient_info.get('fbs') == 1 else 'No'}
- **Resting ECG:** {patient_info.get('restecg')}
- **Max Heart Rate Achieved:** {patient_info.get('thalach')} bpm
- **Exercise-Induced Angina:** {'Yes' if patient_info.get('exang') == 1 else 'No'}
- **ST Depression (oldpeak):** {patient_info.get('oldpeak')}
- **Slope of ST Segment:** {patient_info.get('slope')}
- **Major Vessels (ca):** {patient_info.get('ca')}
- **Thalassemia:** {patient_info.get('thal')}
- **Patient Symptoms / Notes:** {patient_info.get('symptoms') or 'None reported'}

---

## 2. Machine Learning Diagnostic Assessment
- **Algorithm:** Random Forest Classifier (Cleveland Cohort Validation)
- **Calculated Cardiovascular Risk:** {cur_risk}%
- **Risk Classification:** {cur_label}

---

## 3. DeepSeek-V3 Clinical Synthesis & Guidance
{cur_ai}

---

## 4. Medical Disclaimer
This report is generated automatically by an AI algorithm for educational and informational purposes only.
It does NOT constitute medical advice, diagnosis, or treatment. Always discuss these findings with a licensed healthcare provider.
"""

        st.download_button(
            label="📥 Download Health Summary Report (.md)",
            data=report_content,
            file_name=f"health_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Dataset Explorer (Expandable)
    # -------------------------------------------------------------------------
    with st.expander("🔍 Explore Benchmark Clinical Dataset"):
        st.markdown(f"**Dataset Overview ({len(df)} records):**")
        st.dataframe(df.head(10), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Dataset Summary Statistics:**")
            st.dataframe(df[["age", "trestbps", "chol", "thalach", "oldpeak"]].describe(), use_container_width=True)
        with col_b:
            st.write("**Target Distribution:**")
            target_counts = df["target"].value_counts().rename({0: "Healthy (0)", 1: "Heart Disease (1)"})
            st.bar_chart(target_counts)

    # Bottom Medical Disclaimer Callout
    st.markdown("---")
    st.info(
        "⚕️ **Professional Healthcare Reminder:** Do not alter medications, lifestyle regimens, "
        "or clinical treatments based solely on AI predictions. Contact your cardiologist or primary care physician for a personalized evaluation."
    )


if __name__ == "__main__":
    main()
