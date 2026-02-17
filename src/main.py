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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------
# 1. Encode categorical columns
# -------------------------
le = LabelEncoder()

for col in ["weather", "time_of_day", "road_type", "lighting", "traffic", "severity"]:
    data[col] = le.fit_transform(data[col])

print("\nEncoded Dataset:")
print(data.head())

# -------------------------
# 2. Split features and target
# -------------------------
X = data.drop("severity", axis=1)  # input features
y = data["severity"]               # target variable

# -------------------------
# 3. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# -------------------------
# 4. Train a model
# -------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 5. Make predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 6. Evaluate the model
# -------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
