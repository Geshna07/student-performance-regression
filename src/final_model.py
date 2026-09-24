import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input features
features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Marks"
]

X = data[features]
y = data["Final_Marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("\nDataset:")
print(f"Total Students: {len(data)}")
print(f"Training Students: {len(X_train)}")
print(f"Testing Students: {len(X_test)}")

print("\nModel: Multiple Linear Regression")

print("\nR² Score:", round(r2, 3))
print("RMSE:", round(rmse, 3))

print("\nRegression Equation:")
print(
    f"Final_Marks = {model.intercept_:.2f}"
    f" + ({model.coef_[0]:.2f} × Study_Hours)"
    f" + ({model.coef_[1]:.2f} × Attendance)"
    f" + ({model.coef_[2]:.2f} × Assignment_Score)"
    f" + ({model.coef_[3]:.2f} × Previous_Marks)"
)

print("\nFeature Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.3f}")

print("\n" + "=" * 60)