import os
import glob
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from google import genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. LOAD ENV + INIT GEMINI

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

print("Gemini client initialized")

# 2. LOAD DATASET

folder_path = r"C:\Users\Osh Gupta\Downloads\bank_marketing"

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found")

data_file = max(csv_files, key=os.path.getsize)
print(f"Loading dataset: {data_file}")

df = pd.read_csv(data_file, sep=";")
print(f"Dataset shape: {df.shape}")

# 3. DATA LEAKAGE CHECK

leaky_features = []

if "duration" in df.columns:
    leaky_features.append("duration")

print(f"Identified leaky features: {leaky_features}")

# 4. FEATURE / TARGET SPLIT

target = "y"
X = df.drop(columns=[target] + leaky_features)
y = df[target].map({"yes": 1, "no": 0})

# 5. FEATURE TYPES

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# 6. PREPROCESSING PIPELINE

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 7. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. MODEL PIPELINE

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# 9. TRAIN MODEL

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Accuracy (Leakage-Free): {accuracy:.4f}")

# 10. FEATURE IMPORTANCE (APPROX via COEFS)

feature_names = (
    list(numeric_features)
    + list(
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
)

coefficients = model.named_steps["classifier"].coef_[0]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(coefficients)
}).sort_values(by="importance", ascending=False)

top_features = importance_df.head(5)

# 11. GEMINI EXPLANATION (ANALYSIS AGENT)

prompt = f"""
We trained a leakage-free logistic regression model to predict term deposit subscription.

Key details:
- Leaky feature 'duration' was removed
- Final accuracy: {accuracy:.3f}
- Top features influencing prediction:

{top_features.to_string(index=False)}

Explain:
1. Why removing 'duration' was necessary
2. Why accuracy dropped compared to leaky models
3. What business insights Payment Status provides
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n GEMINI ANALYSIS REPORT\n")
print(response.text)

