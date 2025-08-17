import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

st.title("🧠 Smart Health Companion")
st.write("Upload your health data and get insights!")

st.write(df.head())

sns.histplot(df["chol"], kde=True)
st.pyplot(plt)
