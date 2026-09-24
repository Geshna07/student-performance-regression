import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 60)

# Input variables
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Marks"
    ]
]

# Target variable
y = data["Final_Marks"]

# Create regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
predicted_marks = model.predict(X)

# Calculate evaluation metrics
r2 = r2_score(y, predicted_marks)

rmse = np.sqrt(
    mean_squared_error(y, predicted_marks)
)

# Display intercept
print("\nIntercept:")
print(f"{model.intercept_:.3f}")

# Display coefficients
print("\nRegression Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.3f}")

# Display equation
print("\nRegression Equation:")

print(
    f"Final_Marks = {model.intercept_:.2f}"
    f" + ({model.coef_[0]:.2f} × Study_Hours)"
    f" + ({model.coef_[1]:.2f} × Attendance)"
    f" + ({model.coef_[2]:.2f} × Assignment_Score)"
    f" + ({model.coef_[3]:.2f} × Previous_Marks)"
)

# Display evaluation
print("\nModel Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Create prediction table
results = pd.DataFrame({
    "Student_ID": data["Student_ID"],
    "Actual_Final_Marks": y,
    "Predicted_Final_Marks": predicted_marks
})

print("\nFirst 10 Predictions:")
print(results.head(10).round(2))

# Actual vs Predicted graph
plt.figure(figsize=(9, 6))

plt.scatter(
    y,
    predicted_marks,
    label="Students"
)

# Perfect prediction reference line
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")

plt.title(
    "Multiple Linear Regression: Actual vs Predicted Marks"
)

plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "graphs/multiple_regression.png",
    dpi=300
)

plt.show()

print("\nGraph saved as:")
print("graphs/multiple_regression.png")

print("\n" + "=" * 60)
print("MULTIPLE LINEAR REGRESSION COMPLETED")
print("=" * 60)