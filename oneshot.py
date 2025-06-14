import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
submission_df = pd.read_csv('dataset/sample_submission.csv')

train_df = pd.read_csv('dataset/train.csv')
print("Available columns in train.csv:", train_df.columns.tolist())

features = ['SqFeet', 'Bedrooms'] 
target = 'Price'

X = train_df[features]
y = train_df[target]


X = train_df[features]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
mse = mean_squared_error(y_val, val_preds)
print(f"Validation MSE: {mse:.2f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

X_test = test_df[features]
test_preds = model.predict(X_test)

submission_df[target] = test_preds
submission_df.to_csv('my_submission.csv', index=False)
print("✅ Submission saved to 'my_submission.csv'")
