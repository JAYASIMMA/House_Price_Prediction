import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load model and data ===
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")
st.write("Upload your data and predict house prices using a trained model.")

# Load model
model = joblib.load("zero_shot_model.pkl")  # or 'xgboost_model.pkl'
train_df = pd.read_csv("dataset/train.csv")

# --- Sidebar: User Input ---
st.sidebar.header("Enter House Details")
numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).drop(columns=["SalePrice", "Id"]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

user_input = {}
st.sidebar.subheader("Numerical Features")
for col in numerical_cols:
    min_val = train_df[col].min()
    max_val = train_df[col].max()
    mean_val = train_df[col].mean()
    user_input[col] = st.sidebar.slider(col, float(min_val), float(max_val), float(mean_val))

st.sidebar.subheader("Categorical Features")
for col in categorical_cols:
    options = train_df[col].dropna().unique().tolist()
    user_input[col] = st.sidebar.selectbox(col, options)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sale Price: ${prediction:,.2f}")

# === Visualization ===
st.header("📊 Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Histogram", "Correlation Heatmap", "Feature Importance"])

with tab1:
    st.subheader("Histogram of House Sale Prices")
    plt.figure(figsize=(10, 4))
    sns.histplot(train_df["SalePrice"], kde=True, color="skyblue", bins=30)
    plt.xlabel("Sale Price")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())

with tab2:
    st.subheader("Correlation Heatmap (Top 10 features)")
    corr = train_df.corr(numeric_only=True)
    top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(11).index
    plt.figure(figsize=(10, 6))
    sns.heatmap(train_df[top_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt.gcf())

with tab3:
    st.subheader("Feature Importance (for XGBoost only)")

    # Replace 'regressor' with the actual step name (e.g., 'model')
    regressor_step_name = [step for step in model.named_steps if hasattr(model.named_steps[step], "feature_importances_")]
    
    if regressor_step_name:
        reg = model.named_steps[regressor_step_name[0]]

        # Get categorical feature names
        ohe = model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        cat_features = ohe.get_feature_names_out(categorical_cols)
        feature_names = list(numerical_cols) + list(cat_features)

        importances = reg.feature_importances_
        top_idx = np.argsort(importances)[-10:]

        plt.figure(figsize=(8, 4))
        plt.barh(np.array(feature_names)[top_idx], importances[top_idx], color='teal')
        plt.xlabel("Importance Score")
        plt.title("Top 10 Important Features")
        st.pyplot(plt.gcf())
    else:
        st.info("Feature importance is available only for tree-based models like XGBoost.")
