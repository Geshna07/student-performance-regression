import numpy as np
import pandas as pd

# Make the results reproducible
np.random.seed(42)

# Number of students
num_students = 100

# Student IDs
student_ids = [
    f"S{i:03d}"
    for i in range(1, num_students + 1)
]

# Generate realistic student data
study_hours = np.round(
    np.random.uniform(2, 10, num_students),
    1
)

attendance = np.round(
    np.random.uniform(60, 100, num_students),
    1
)

assignment_score = np.round(
    np.random.uniform(50, 95, num_students),
    1
)

previous_marks = np.round(
    np.random.uniform(45, 90, num_students),
    1
)

# Generate final marks with some realistic variation
noise = np.random.normal(0, 4, num_students)

final_marks = (
    20
    + 2.5 * study_hours
    + 0.20 * attendance
    + 0.25 * assignment_score
    + 0.35 * previous_marks
    + noise
)

# Keep marks between 35 and 100
final_marks = np.clip(
    final_marks,
    35,
    100
)

final_marks = np.round(
    final_marks,
    1
)

# Create DataFrame
data = pd.DataFrame({
    "Student_ID": student_ids,
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Previous_Marks": previous_marks,
    "Final_Marks": final_marks
})

# Save the dataset
data.to_csv(
    "data/student_data.csv",
    index=False
)

print("Dataset created successfully!")
print(f"Number of students: {len(data)}")

print("\nFirst 10 students:")
print(data.head(10))

print("\nDataset saved as:")
print("data/student_data.csv")