import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# === Step 1: Load Data ===
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
sample_submission = pd.read_csv("dataset/sample_submission.csv")


# === Step 2: Prepare Inputs ===
X = train_df.drop(columns=["SalePrice", "Id"])
y = train_df["SalePrice"]
X_test = test_df.drop(columns=["Id"])
test_ids = test_df["Id"]

# === Step 3: Identify Column Types ===
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# === Step 4: Define Preprocessing Pipelines ===
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# === Step 5: Define the Zero-Shot Model Pipeline ===
zero_shot_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === Step 6: Fit on Full Data (Zero-Shot) ===
zero_shot_model.fit(X, y)

# === Step 7: Evaluate (use train-validation split only for reporting) ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
zero_shot_preds = zero_shot_model.predict(X_val)
r2 = r2_score(y_val, zero_shot_preds)
print(f"✅ Zero-Shot Model R² Score: {r2:.4f}")

# === Step 8: Save the Trained Model ===
joblib.dump(zero_shot_model, "zero_shot_model.pkl")
print("📦 Model saved to 'zero_shot_model.pkl'")

# === (Optional) Step 9: Generate Submission File ===
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": zero_shot_model.predict(X_test)
})
submission.to_csv("submission.csv", index=False)
print("📄 Submission file 'submission.csv' created!")
