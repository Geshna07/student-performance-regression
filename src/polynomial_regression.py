import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("POLYNOMIAL REGRESSION")
print("=" * 60)

# Input variable
X = data[["Study_Hours"]]

# Target variable
y = data["Final_Marks"]

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

# Create regression model
model = LinearRegression()

# Train the model
model.fit(X_poly, y)

# Make predictions
predicted_marks = model.predict(X_poly)

# Calculate metrics
r2 = r2_score(y, predicted_marks)

rmse = np.sqrt(
    mean_squared_error(y, predicted_marks)
)

# Get coefficients
intercept = model.intercept_
coefficients = model.coef_

print("\nPolynomial Regression Equation:")

print(
    f"Final_Marks = {intercept:.2f}"
    f" + ({coefficients[1]:.2f} × Study_Hours)"
    f" + ({coefficients[2]:.2f} × Study_Hours²)"
)

print("\nModel Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Prediction table
results = pd.DataFrame({
    "Student_ID": data["Student_ID"],
    "Study_Hours": data["Study_Hours"],
    "Actual_Final_Marks": y,
    "Predicted_Final_Marks": predicted_marks
})

print("\nFirst 10 Predictions:")
print(results.head(10).round(2))

# Create smooth curve for graph
x_range = np.linspace(
    data["Study_Hours"].min(),
    data["Study_Hours"].max(),
    200
).reshape(-1, 1)

x_range_poly = poly.transform(x_range)

y_range = model.predict(x_range_poly)

# Plot actual data
plt.figure(figsize=(9, 6))

plt.scatter(
    data["Study_Hours"],
    y,
    label="Actual Data"
)

plt.plot(
    x_range,
    y_range,
    label="Polynomial Regression Curve"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")

plt.title(
    "Polynomial Regression: Study Hours vs Final Marks"
)

plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "graphs/polynomial_regression.png",
    dpi=300
)

plt.show()

print("\nGraph saved as:")
print("graphs/polynomial_regression.png")

print("\n" + "=" * 60)
print("POLYNOMIAL REGRESSION COMPLETED")
print("=" * 60)