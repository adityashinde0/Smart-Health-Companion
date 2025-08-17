import streamlit as st
import openai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up OpenAI API (DeepSeek)
try:
    with open("my_api_key.txt", "r") as f:
        openai.api_key = f.read().strip()
except Exception:
    st.warning("API key not found in secrets. Please add it to .streamlit/secrets.toml.")
    st.stop()
openai.base_url = "https://inference.cloudrift.ai/v1"

# Load Dataset
try:
    df = pd.read_csv("heart.csv")
except Exception as e:
    st.error(f"Could not load heart.csv: {e}")
    st.stop()

# App Title
st.title("🩺 Smart Health Companion")
st.markdown("Enter your details to get a heart health analysis and prediction.")

# User Inputs
with st.form("user_input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    symptoms = st.text_area("Describe any symptoms you're experiencing")
    heart_rate = st.slider("Your current heart rate (bpm)", 40, 180, 72)
    submitted = st.form_submit_button("Analyze My Health")

# Show Dataset Insights
st.subheader("Dataset Insights")
st.write("Sample of the dataset:")
st.dataframe(df.head())

st.write("Cholesterol Distribution in Dataset:")
fig1, ax1 = plt.subplots()
ax1.hist(df["chol"], bins=30, color="skyblue", edgecolor="black")
ax1.set_xlabel("Cholesterol (mg/dL)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Heart Rate Visualization
st.subheader("Your Heart Rate Pattern")
x = np.linspace(0, 10, 100)
y = heart_rate + 10 * np.sin(x)
fig2, ax2 = plt.subplots()
ax2.plot(x, y, label="Heart Rate Pattern")
ax2.axhline(y=100, color='r', linestyle='--', label="High Risk Threshold")
ax2.set_xlabel("Time")
ax2.set_ylabel("Heart Rate (bpm)")
ax2.legend()
st.pyplot(fig2)

# AI Prediction
if submitted:
    prompt = f"""
You are a smart health companion. Analyze the following health data and predict possible heart-related diseases or risks, and suggest next steps.

Age: {age}
Sex: {sex}
Chest Pain Type: {cp}
Resting Blood Pressure: {trestbps} mmHg
Cholesterol: {chol} mg/dL
Fasting Blood Sugar > 120 mg/dL: {fbs}
Resting ECG: {restecg}
Max Heart Rate Achieved: {thalach} bpm
Exercise Induced Angina: {exang}
ST depression: {oldpeak}
Slope of ST segment: {slope}
Number of major vessels: {ca}
Thalassemia: {thal}
Symptoms: {symptoms}

Based on the above and the heart.csv dataset, predict the most likely heart disease (if any) and give advice.
"""
    with st.spinner("Analyzing your health..."):
        try:
            response = openai.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            prediction = ""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                    prediction += chunk.choices[0].delta.content
            st.success("Prediction complete!")
            st.markdown("### AI Prediction & Advice")
            st.write(prediction)
        except Exception as e:
            st.error(f"API call failed: {e}")
