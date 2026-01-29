import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. PAGE CONFIG & SETUP
st.set_page_config(page_title="Bank AI Analyst", layout="wide")
load_dotenv()

# Secure API Key Check
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# 2. CACHED MODEL TRAINING FUNCTION
@st.cache_resource
def run_ml_pipeline(df):
    # Step: Identify Leaky Features
    leaky = ["duration"] if "duration" in df.columns else []
    
    # Step: Feature/Target Split
    X = df.drop(columns=["y"] + leaky)
    y = df["y"].map({"yes": 1, "no": 0})
    
    # Step: Preprocessing
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    
    # Step: Pipeline & Train
    model_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model_pipe.fit(X_train, y_train)
    
    # Accuracy & Importance
    acc = accuracy_score(y_test, model_pipe.predict(X_test))
    
    # Extract Feature Importance
    cat_names = model_pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_names)
    coefs = model_pipe.named_steps["classifier"].coef_[0]
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(coefs)
    }).sort_values(by="importance", ascending=False).head(10)
    
    return acc, importance_df, leaky

# 3. STREAMLIT UI
st.title("Bank Marketing AI Analyst")
st.markdown("Predicting term deposit subscriptions using Leakage-Free Machine Learning.")

uploaded_file = st.file_uploader("Upload bank-full.csv", type="csv")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file, sep=";")
    
    # Run Pipeline
    with st.spinner("🧠 Training model and analyzing patterns..."):
        acc, importance, leaky_found = run_ml_pipeline(df)
    
    # Layout: Top Row Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{acc:.2%}")
    with col2:
        st.warning(f"Dropped Leaky Features: {', '.join(leaky_found)}")

    # Layout: Feature Importance Chart
    st.subheader("Top Decision Factors")
    st.bar_chart(importance.set_index("feature"))

    # Layout: Gemini Analysis
    st.subheader("Gemini Business Report")
    
    prompt = f"""
    Explain this Bank Marketing ML result:
    - Accuracy: {acc:.2%}
    - Removed Leakage: {leaky_found}
    - Top Features: {importance.to_dict()}
    
    Provide 3 concise business insights for the marketing team.
    """

    with st.spinner("Generating AI Insights..."):
        # Force V1 Stable endpoint for 2026 compatibility
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        st.markdown(response.text)
    
    st.balloons()