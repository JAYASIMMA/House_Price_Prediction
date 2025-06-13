import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# === Step 1: Load Datasets ===
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# === Step 2: Split Features and Target ===
X = train_df.drop(columns=["SalePrice", "Id"])
y = train_df["SalePrice"]
X_test = test_df.drop(columns=["Id"])
test_ids = test_df["Id"]

# === Step 3: Feature Groups ===
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# === Step 4: Preprocessing Pipelines ===
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# === Step 5: Create One-Shot Pipeline ===
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# === Step 6: Train-Test Split for Local Validation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# === Step 7: Evaluate ===
val_preds = model_pipeline.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"📊 R² Score on Validation Set: {r2:.4f}")

# === Step 8: Train on Full Data and Predict on Test ===
model_pipeline.fit(X, y)
test_preds = model_pipeline.predict(X_test)

# === Step 9: Save Model ===
joblib.dump(model_pipeline, "oneshot_linear_model.pkl")
print("✅ Saved model to 'oneshot_linear_model.pkl'")

# === Step 10: Create Submission ===
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_preds
})
submission.to_csv("submission_oneshot_linear.csv", index=False)
print("📄 Submission file 'submission_oneshot_linear.csv' created!")
