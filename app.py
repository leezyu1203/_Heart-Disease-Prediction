from sklearn.pipeline import Pipeline
import streamlit as st
import joblib
import pandas as pd

try:
  model = joblib.load('prediction_model.pkl')
except FileNotFoundError:
  st.error("Model file 'pridiction_model.pkl' not found. Please ensure it's in the correct directory.")
  st.stop() # Stop the app if model isn't found

df_in = pd.DataFrame()

@st.dialog("Please complete the form")
def hasNullDialog():
  st.write("Incomplete infomation: ", df_in.columns[df_in.isnull().any()])
  return

st.set_page_config(
  page_title="Heart Disease Prediction",
  layout="centered"
)

st.markdown("""
  <style>
    .appview-container {
      background-color: #faf8f1;
    }
    .stAppHeader {
      background-color: #fac9c9;
    }
  </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction")

with st.form("form", enter_to_submit=False, border=False):
  col1, col2 = st.columns(2)
  with col1: 
    df_in["Sex"] = [st.selectbox(
      "Sex",
      ("Female", "Male"),
      index=None,
      placeholder="Select your gender"
    )]
    df_in["AgeCategory"] = [st.selectbox(
      "Age Category",
      ("18-24", "25-29", "30-34", "35-39","40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"),
      index=None,
      placeholder="Select your age category"
    )]
  with col2:
    df_in["Race"] = [st.selectbox(
      "Race",
      ("White", "Black", "Asian", "American Indian/Alaskan Native", "Hispanic", "Other"),
      index=None,
      placeholder="Select your race"
    )]
    df_in["BMI"] = [st.number_input("BMI", 10.0, 50.0)]
  df_in["SleepTime"] = [st.slider("On average, how many hours of sleep do you get in a 24-hour period?", 1, 12, 8)]

  st.subheader("Habits")
  df_in["Smoking"] = [st.toggle("Have you smoked at least once in the past 30 days?")]
  df_in["AlcoholDrinking"] = [st.toggle("Have you had at least one drink of alcohol in the past 30 days?")]
  df_in["PhysicalActivity"] = [st.toggle("Have you participated in any physical activities or exercises in the past 30 days?")]

  st.subheader("Health Status and History")
  df_in["DiffWalking"] = [st.toggle("Do you have serious difficulty walking or climbing stairs?")]
  df_in["GenHealth"] = [st.selectbox(
    "Would you say that in general your health is...",
    ("Poor", "Fair", "Good", "Very good", "Excellent"),
    index=None,
    placeholder="Select your general health status",
  )]
  col3, col4 = st.columns(2)
  with col3:
    df_in["PhysicalHealth"] = [st.number_input("How many days during the past 30 days was your physical health not good?", 0)]
    df_in["Stroke"] = [st.toggle("Do you have a history of stroke?")]
    df_in["KidneyDisease"] = [st.toggle("Do you currently suffer from or have a history of kidney disease?")]
  with col4: 
    df_in["MentalHealth"] = [st.number_input("How many days during the past 30 days was your mental health not good?", 0)]
    df_in["Asthma"] = [st.toggle("Do you have asthma?")]
    df_in['SkinCancer'] = [st.toggle("Do you currently suffer from or have a history of skin cancer?")]
  df_in["Diabetic"] = [st.selectbox(
    "Do you currently suffer from diabetes?",
    ("Yes", "No", "No, borderline diabetes", "Yes (during pragnancy)"),
    index=None,
    placeholder="Diabetic"
  )]

  submit = st.form_submit_button("Submit")
  if submit:
    if df_in.isnull().any().any():
      hasNullDialog()
    else:
      true_false_cols = ['Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'DiffWalking', 'Stroke', 'KidneyDisease', 'Asthma', 'SkinCancer']
      df_in[true_false_cols] = df_in[true_false_cols].replace([True, False], ["Yes", "No"])
      try:
        prediction = model.predict(df_in)[0]
        prediction_proba = model.predict_proba(df_in)[0]
        if prediction == 1:
          st.error("**Prediction:** Heart Disease")
        else:
          st.success("**Prediction:** No Disease")
        st.write(f"Probability of No Disease: {prediction_proba[0]:.4f}")
        st.write(f"Probability of Heart Disease: {prediction_proba[1]:.4f}")
      except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure your inputs are valid and match the expected format.")


