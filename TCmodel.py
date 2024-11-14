import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Assume it's in CSV format)
file_path = 'TC/network_traffic_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print(data.head())

# Basic preprocessing
# 1. Handle missing values (if any)
data = data.dropna()

# 2. Convert categorical features (like 'Protocol', 'SourceIP', 'DestinationIP') to numerical using LabelEncoder

# Encode 'Protocol' column
label_encoder = LabelEncoder()
data['Protocol'] = label_encoder.fit_transform(data['Protocol'])

# Split IP addresses into four parts and treat each part as a separate feature
def encode_ip(ip):
    return list(map(int, ip.split('.')))

data[['SourceIP_1', 'SourceIP_2', 'SourceIP_3', 'SourceIP_4']] = data['SourceIP'].apply(encode_ip).apply(pd.Series)
data[['DestinationIP_1', 'DestinationIP_2', 'DestinationIP_3', 'DestinationIP_4']] = data['DestinationIP'].apply(encode_ip).apply(pd.Series)

# Drop original IP columns
data = data.drop(['SourceIP', 'DestinationIP'], axis=1)

# 3. Split the data into features (X) and target (y)
X = data.drop('Label', axis=1)  # Drop the target column
y = data['Label']  # Target variable (Benign: 0, Malicious: 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance (for Random Forest)
feature_importances = model.feature_importances_
features = X.columns

# Plot the importance of features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()