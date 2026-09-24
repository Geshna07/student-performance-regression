import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("SIMPLE LINEAR REGRESSION")
print("=" * 60)

# Input variable
X = data[["Study_Hours"]]

# Target variable
y = data["Final_Marks"]

# Create regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
predicted_marks = model.predict(X)

# Get model parameters
slope = model.coef_[0]
intercept = model.intercept_

# Calculate evaluation metrics
r2 = r2_score(y, predicted_marks)

rmse = np.sqrt(
    mean_squared_error(y, predicted_marks)
)

print("\nRegression Equation:")
print(
    f"Final_Marks = {intercept:.2f} + "
    f"({slope:.2f} × Study_Hours)"
)

print("\nModel Results:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Show sample predictions
results = pd.DataFrame({
    "Study_Hours": data["Study_Hours"],
    "Actual_Final_Marks": y,
    "Predicted_Final_Marks": predicted_marks
})

print("\nFirst 10 Predictions:")
print(results.head(10).round(2))

# Create scatter plot
plt.figure(figsize=(9, 6))

plt.scatter(
    data["Study_Hours"],
    y,
    label="Actual Data"
)

plt.plot(
    data["Study_Hours"],
    predicted_marks,
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Simple Linear Regression: Study Hours vs Final Marks")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    "graphs/simple_linear_regression.png",
    dpi=300
)

plt.show()

print("\nGraph saved as:")
print("graphs/simple_linear_regression.png")

print("\n" + "=" * 60)
print("SIMPLE LINEAR REGRESSION COMPLETED")
print("=" * 60)