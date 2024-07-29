import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
x = iris.data[:, :2]  # Use only the first two features for easier visualization
y = iris.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN classifier
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(x_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNN model accuracy:", accuracy)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the training data points
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired, label='Training', edgecolor='k')

# Plot the test data points
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Paired, label='Test', marker='x', s=100)

# Add a legend
plt.legend()

# Add axis labels
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Add a title
plt.title('KNN Results Visualization on the Iris Dataset')

plt.show()
