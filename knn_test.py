import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

# Load the dataset from the CSV file
data = pd.read_csv('Social_Network_Ads.csv')

# Separate the features (X) and the target (y) from the dataset
X = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling: Standardize the features to have mean=0 and variance=1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an object of KNN (KNearestNeighbors) classifier with k=5
knn = KNearestNeighbors(k=5)

# Fit the KNN model to the training data
knn.fit(X_train, y_train)

# Function to predict whether a person will purchase based on age and salary inputs
def predict_new():
    age = int(input("Enter the age: "))
    salary = int(input("Enter the salary: "))
    X_new = np.array([[age], [salary]]).reshape(1, 2)

    # Scale the new input using the same scaler used for training data
    X_new = scaler.transform(X_new)

    # Predict the class label (0: Will not purchase, 1: Will purchase)
    result = knn.predict(X_new)

    # Print the prediction result
    if result == 0:
        print("Will not purchase")
    else:
        print("Will purchase")

# Call the function to predict new data points
predict_new()
