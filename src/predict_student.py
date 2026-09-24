import pandas as pd
from sklearn.linear_model import LinearRegression
import os


# Load dataset
data = pd.read_csv("data/student_data.csv")


# Input variables
features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Marks"
]

X = data[features]
y = data["Final_Marks"]


# Train Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)


print("=" * 60)
print("STUDENT FINAL MARK PREDICTION SYSTEM")
print("=" * 60)

print("\nEnter student details:")


# Get user input
study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
assignment_score = float(input("Assignment Score: "))
previous_marks = float(input("Previous Marks: "))


# Create input DataFrame
student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Assignment_Score": [assignment_score],
    "Previous_Marks": [previous_marks]
})


# Predict final marks
prediction = model.predict(student)[0]

# Keep prediction between 0 and 100
prediction = max(0, min(100, prediction))


# Performance category
if prediction >= 90:
    performance = "Excellent"
elif prediction >= 75:
    performance = "Good"
elif prediction >= 60:
    performance = "Average"
else:
    performance = "Needs Improvement"


# Display result
print("\n" + "=" * 60)
print(f"Predicted Final Marks: {prediction:.2f}")
print(f"Predicted Performance: {performance}")
print("=" * 60)


# Save prediction
os.makedirs("results", exist_ok=True)

prediction_data = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Assignment_Score": [assignment_score],
    "Previous_Marks": [previous_marks],
    "Predicted_Final_Marks": [round(prediction, 2)],
    "Predicted_Performance": [performance]
})


file_path = "results/predictions.csv"

# Add to existing file if it exists
if os.path.exists(file_path):
    prediction_data.to_csv(
        file_path,
        mode="a",
        header=False,
        index=False
    )
else:
    prediction_data.to_csv(
        file_path,
        index=False
    )


print("\nPrediction saved to:")
print(file_path)

print("=" * 60)