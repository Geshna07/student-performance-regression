import pandas as pd

# Load the dataset
data = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("STUDENT PERFORMANCE DATA ANALYSIS")
print("=" * 60)

# 1. Number of rows and columns
print("\n1. Dataset Shape")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# 2. Column names
print("\n2. Column Names")
print(data.columns.tolist())

# 3. First five records
print("\n3. First Five Records")
print(data.head())

# 4. Data information
print("\n4. Dataset Information")
data.info()

# 5. Missing values
print("\n5. Missing Values")
print(data.isnull().sum())

# 6. Duplicate rows
print("\n6. Duplicate Rows")
print(data.duplicated().sum())

# 7. Statistical summary
print("\n7. Statistical Summary")
print(data.describe())

# 8. Minimum values
print("\n8. Minimum Values")
print(data.min(numeric_only=True))

# 9. Maximum values
print("\n9. Maximum Values")
print(data.max(numeric_only=True))

print("\n" + "=" * 60)
print("DATA VALIDATION COMPLETED")
print("=" * 60)