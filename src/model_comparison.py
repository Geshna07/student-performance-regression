import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("REGRESSION MODEL COMPARISON")
print("=" * 60)

# Target variable
y = data["Final_Marks"]


# --------------------------------------------------
# 1. SIMPLE LINEAR REGRESSION
# --------------------------------------------------

X_simple = data[["Study_Hours"]]

simple_model = LinearRegression()
simple_model.fit(X_simple, y)

simple_predictions = simple_model.predict(X_simple)

simple_r2 = r2_score(y, simple_predictions)

simple_rmse = np.sqrt(
    mean_squared_error(y, simple_predictions)
)


# --------------------------------------------------
# 2. POLYNOMIAL REGRESSION
# --------------------------------------------------

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X_simple)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_predictions = poly_model.predict(X_poly)

poly_r2 = r2_score(y, poly_predictions)

poly_rmse = np.sqrt(
    mean_squared_error(y, poly_predictions)
)


# --------------------------------------------------
# 3. MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

X_multiple = data[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Marks"
    ]
]

multiple_model = LinearRegression()
multiple_model.fit(X_multiple, y)

multiple_predictions = multiple_model.predict(X_multiple)

multiple_r2 = r2_score(y, multiple_predictions)

multiple_rmse = np.sqrt(
    mean_squared_error(y, multiple_predictions)
)


# --------------------------------------------------
# CREATE COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Polynomial Regression",
        "Multiple Linear Regression"
    ],
    "R2_Score": [
        simple_r2,
        poly_r2,
        multiple_r2
    ],
    "RMSE": [
        simple_rmse,
        poly_rmse,
        multiple_rmse
    ]
})

# Round values
comparison["R2_Score"] = comparison["R2_Score"].round(3)
comparison["RMSE"] = comparison["RMSE"].round(3)

print("\nModel Comparison:")
print(comparison.to_string(index=False))


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nResults saved as:")
print("results/model_comparison.csv")


# --------------------------------------------------
# CREATE R2 COMPARISON GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Model"],
    comparison["R2_Score"]
)

plt.xlabel("Regression Model")
plt.ylabel("R² Score")

plt.title("Comparison of Regression Models")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    "graphs/model_comparison_r2.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# CREATE RMSE COMPARISON GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.xlabel("Regression Model")
plt.ylabel("RMSE")

plt.title("RMSE Comparison of Regression Models")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    "graphs/model_comparison_rmse.png",
    dpi=300
)

plt.show()


print("\nGraphs saved:")
print("graphs/model_comparison_r2.png")
print("graphs/model_comparison_rmse.png")

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED")
print("=" * 60)