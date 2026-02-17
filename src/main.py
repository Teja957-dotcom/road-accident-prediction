import pandas as pd

print("🚦 Road Accident Severity Prediction Project Started!")

# Load dataset
data = pd.read_csv("data/accidents.csv")

print("\n✅ Dataset Loaded Successfully!\n")

# Show first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nBasic Statistics:")
print(data.describe(include="all"))
