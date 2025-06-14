import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
sample_submission = pd.read_csv("dataset/sample_submission.csv")

X = train_df.drop(columns=["SalePrice", "Id"])
y = train_df["SalePrice"]
X_test = test_df.drop(columns=["Id"])
test_ids = test_df["Id"]

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_pipeline.fit(X_train, y_train)

val_preds = xgb_pipeline.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"✅ XGBoost R² Score on Validation Set: {r2:.4f}")

joblib.dump(xgb_pipeline, "xgboost_model.pkl")
print("📦 Model saved to 'xgboost_model.pkl'")

submission_preds = xgb_pipeline.predict(X_test)
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": submission_preds
})
submission.to_csv("submission_xgboost.csv", index=False)
print("📄 Submission file 'submission_xgboost.csv' created!")
