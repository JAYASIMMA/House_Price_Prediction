# 🏠 House Price Prediction Web App

This project is a machine learning-based web application built with **Streamlit** for predicting house prices. It uses **Linear Regression** and optionally **XGBoost**, trained on housing data with preprocessing steps included in a pipeline.

---

## 📁 Project Structure

```

House\_Price\_Prediction/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample\_submission.csv
│
├── oneshot\_linear.py             # Train linear regression model
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample\_submission.csv
│
├── train\_xgboost.py             # Train XGBoost model (optional)
├── app.py                       # Streamlit frontend app
├── zero\_shot\_model.pkl          # Trained model file (e.g., Linear or XGBoost)
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample\_submission.csv
│
├── submission\_oneshot\_linear.csv # Generated submission file
├── requirements.txt
└── README.md

````

---

## 🚀 Features

- Upload custom house data and get predicted price
- Interactive sidebar for input features
- Visualizations:
  - Histogram of Sale Prices
  - Correlation Heatmap
  - Feature Importance (for XGBoost models)
- Preprocessing includes:
  - Missing value handling
  - One-hot encoding
  - Model serialization using `joblib`

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

**requirements.txt**

```txt
pandas
scikit-learn
joblib
xgboost
streamlit
matplotlib
seaborn
```

---

## 🧠 Training the Model

### ➤ Linear Regression

```bash
python oneshot_linear.py
```

This will:

* Train a model on `train.csv`
* Save predictions to `submission_oneshot_linear.csv`
* Export trained model as `oneshot_linear_model.pkl`

### ➤ XGBoost (optional)

```bash
python train_xgboost.py
```

Exports model as `xgboost_model.pkl`

---

## 💻 Running the Web App

To launch the Streamlit frontend:

```bash
streamlit run app.py
```

Make sure your trained model (e.g., `zero_shot_model.pkl`) is present in the directory.

---

## 🔍 Example Inputs

* **Numerical Inputs:** LotArea, GrLivArea, YearBuilt, etc.
* **Categorical Inputs:** Neighborhood, HouseStyle, etc.

---

## 📊 Screenshots
![Screenshot 2025-06-13 180409](https://github.com/user-attachments/assets/876c3e99-fef5-4468-b197-e8620bed368f)
![Screenshot 2025-06-13 180901](https://github.com/user-attachments/assets/daac0b1a-7a55-4fec-8f4d-d28fed5dec2e)
![Screenshot 2025-06-13 185950](https://github.com/user-attachments/assets/ccee2d5b-1709-4ab0-804a-86e5cd2503ec)
![Screenshot 2025-06-13 191924](https://github.com/user-attachments/assets/44be7c36-23db-4a4a-9257-0c54cb2f380d)
![Screenshot 2025-06-13 192013](https://github.com/user-attachments/assets/40209e8e-3834-495f-832f-7b79ac4c2455)
![Screenshot 2025-06-13 192036](https://github.com/user-attachments/assets/2f9e2ede-c2f8-4bf5-bab1-e40132e1857b)
![Screenshot 2025-06-13 192051](https://github.com/user-attachments/assets/30a57817-2f8c-4da5-8caf-4c608cc8458c)
![Screenshot 2025-06-13 192157](https://github.com/user-attachments/assets/d88603e9-be26-4bfd-912b-a29dddd88758)
![Screenshot 2025-06-13 192302](https://github.com/user-attachments/assets/eea59278-63fd-4275-8e2c-18adfef26382)


---

## 🤖 Model Notes

* Linear Regression gives a good baseline performance.
* XGBoost supports feature importance plotting.
* Trained pipeline includes full preprocessing using `ColumnTransformer`.

---

## 🧾 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Jayasimma D**

Feel free to connect on [LinkedIn](https://in.linkedin.com/in/jayasimma-d-4057ab27b) or contribute to this project!
