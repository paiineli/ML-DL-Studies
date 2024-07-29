# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Generating a synthetic dataset for binary classification
x, y = make_blobs(n_samples=50, centers=2, random_state=6)

# Creating the SVM classifier
clf = svm.SVC(kernel='linear')

# Training the SVM model
clf.fit(x, y)

# Plotting the data points
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# Plotting the decision boundary
# Obtaining the coefficients of the hyperplane
w = clf.coef_[0]
b = clf.intercept_[0]

# Generating values for the decision boundary line
x_plot = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
y_plot = - (w[0] * x_plot + b) / w[1]

# Plotting the decision boundary line
plt.plot(x_plot, y_plot, 'k-')

# Plotting the support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
plt.title('SVM Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
