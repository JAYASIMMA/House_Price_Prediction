import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

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

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

val_preds = model_pipeline.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"📊 R² Score on Validation Set: {r2:.4f}")

model_pipeline.fit(X, y)
test_preds = model_pipeline.predict(X_test)

joblib.dump(model_pipeline, "oneshot_linear_model.pkl")
print("✅ Saved model to 'oneshot_linear_model.pkl'")

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_preds
})
submission.to_csv("submission_oneshot_linear.csv", index=False)
print("📄 Submission file 'submission_oneshot_linear.csv' created!")
