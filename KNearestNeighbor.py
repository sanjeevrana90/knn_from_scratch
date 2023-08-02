import operator
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k): 
        # Initialize the KNearestNeighbors object with the value of k (number of neighbors)
        self.k = k

    def fit(self, X_train, y_train):
        # Fit the KNN model to the training data
        self.X_train = X_train
        self.y_train = y_train
        # print("Training Done")

    def predict(self, X_test):
        # Predict the class labels for the given test data using KNN algorithm
        
        # Step 1: Calculate the Euclidean distance between each test data point and all training data points
        distance = {}
        counter = 1

        for i in self.X_train:
            # Calculate Euclidean distance for each training data point and store in the distance dictionary
            distance[counter] = ((X_test[0][0] - i[0])** 2 + (X_test[0][1] - i[1])**2)**1/2
            counter = counter + 1

        # Step 2: Sort the distances in ascending order and select the top k nearest neighbors
        distance = sorted(distance.items(), key = operator.itemgetter(1))
        k_nearest = distance[:self.k]

        # Step 3: Classify the test data point based on the majority class of its k nearest neighbors
        return self.classify(distance=k_nearest)

    def classify(self, distance):
        # Classify the test data point based on the majority class of its k nearest neighbors
        
        # Step 1: Collect the class labels of the k nearest neighbors
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        
        # Step 2: Count the occurrences of each class label and return the most common class
        return Counter(label).most_common()[0][0]
