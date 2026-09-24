import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Select numerical columns
numeric_data = data[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Marks",
        "Final_Marks"
    ]
]

# Calculate correlation matrix
correlation = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation.round(3))

# Correlation with Final Marks
print("\nCorrelation with Final Marks:")
print(
    correlation["Final_Marks"]
    .sort_values(ascending=False)
    .round(3)
)

# Create correlation heatmap
plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

# Add correlation values inside the graph
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap of Student Performance")
plt.tight_layout()

# Save graph
plt.savefig(
    "graphs/correlation_heatmap.png",
    dpi=300
)

plt.show()

print("\nCorrelation heatmap saved as:")
print("graphs/correlation_heatmap.png")

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS COMPLETED")
print("=" * 60)