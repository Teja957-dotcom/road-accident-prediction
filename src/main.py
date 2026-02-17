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

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Plot severity count
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="severity")
plt.title("Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()

# Plot weather vs severity
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="weather", hue="severity")
plt.title("Weather vs Accident Severity")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.show()

# Plot time of day vs severity
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="time_of_day", hue="severity")
plt.title("Time of Day vs Accident Severity")
plt.xlabel("Time of Day")
plt.ylabel("Count")
plt.show()
