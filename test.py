"""
This is test.py file to test the model accuracy.
"""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Loading the model
model = joblib.load('logisticRegressionModel.joblib')

X_test = pd.read_csv('x_test.csv')
y_true = pd.read_csv('y_test.csv')

# Predict using the loaded model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:", accuracy)
