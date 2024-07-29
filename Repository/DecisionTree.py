import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(42)  # For reproducibility
colors = np.random.choice(['Orange', 'Red'], size=90000)
textures = np.random.choice(['Rough', 'Smooth'], size=90000)
fruits = np.random.choice(['Orange', 'Apple'], size=90000)

# Create DataFrame
df = pd.DataFrame({'Color': colors, 'Texture': textures, 'Fruit': fruits})

# Map categorical data to numerical data
df['Color'] = df['Color'].map({'Orange': 0, 'Red': 1})
df['Texture'] = df['Texture'].map({'Rough': 0, 'Smooth': 1})
df['Fruit'] = df['Fruit'].map({'Orange': 0, 'Apple': 1})

# Separate features and target
x = df[['Color', 'Texture']]
y = df['Fruit']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
predictions = model.predict(x_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print('Model accuracy: {:.2f}'.format(accuracy))

# Count samples of each class
class_counts = df['Fruit'].value_counts()

# Bar plot
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['orange', 'red'])
plt.title('Class distribution')
plt.xlabel('Class')
plt.ylabel('Quantity')
plt.xticks([0, 1], ['Orange', 'Apple'], rotation=0)
plt.show()
