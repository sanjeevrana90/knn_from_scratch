# knn_from_scratch
KNearestNeighbors.py:

The KNearestNeighbors.py file contains a Python class KNearestNeighbors that implements the k-nearest neighbors (KNN) algorithm. 
KNN is a simple and effective classification algorithm used for supervised machine learning tasks. The class has three main methods:

__init__(self, k): Constructor method that initializes the KNearestNeighbors object with the value of k, 
which represents the number of nearest neighbors to consider during classification.

fit(self, X_train, y_train): This method is used to train the KNN model. It takes the training data X_train (feature matrix) 
and y_train (target vector) as input and stores them within the object for later use during prediction.

predict(self, X_test): This method is used to predict the class labels for new data points. 
It takes the test data X_test as input and performs the KNN algorithm to find the k nearest neighbors for each test data point in X_test. 
The majority class among the nearest neighbors is then used as the predicted class for each test data point.

classify(self, distance): This method is a helper function used by the predict() method to classify a single test
data point based on its nearest neighbors. It takes the distance list containing the indices of nearest neighbors
and returns the majority class label.

knn_test.py:

The knn_test.py file is a script that demonstrates the usage of the KNearestNeighbors class for predicting whether a person
 will purchase a product based on their age and salary. The script performs the following steps:

Data Loading and Preprocessing: It loads the dataset from a CSV file, separates the features (age and salary) and the target (purchase)
from the dataset, and splits the data into training and testing sets. It also performs feature scaling using StandardScaler.

KNN Model Creation and Training: It creates an instance of the KNearestNeighbors class with k=5, and then trains the KNN model using the training data.

Predicting New Data: It defines a function predict_new() to take age and salary inputs from the user, converts the inputs into a format 
suitable for prediction, scales the new data, and finally predicts whether the person will make a purchase or not using the trained KNN model.

Printing the Prediction: The script then prints the prediction result, indicating whether the person is likely to purchase the product or
not based on the input age and salary.

Overall, knn_test.py demonstrates the practical application of the KNN algorithm by predicting the target class for new data points using
the trained model.

![screenshot](https://github.com/sanjeevrana90/knn_from_scratch/assets/122264554/272703d4-7cd6-4ce0-97d9-a1a166c1e451)
