import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
tv_expenses = np.array([230, 44, 17, 200, 60]).reshape(-1, 1)  # Reshape to a 2D array
radio_expenses = np.array([37, 39, 45, 45, 48]).reshape(-1, 1)
newspaper_expenses = np.array([69, 45, 69, 69, 69]).reshape(-1, 1)
sales = np.array([480, 200, 150, 700, 400])

# List of tuples to iterate over the expense data
expense_data = [("TV", tv_expenses), ("Radio", radio_expenses), ("Newspaper", newspaper_expenses)]

# Create the linear regression model
regression_model = LinearRegression()

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over the expense data
for i, (expense_type, expenses) in enumerate(expense_data):
    # Fit the model to the data
    regression_model.fit(expenses, sales)

    # Model coefficient
    coefficient = regression_model.coef_[0]
    intercept = regression_model.intercept_

    # Plot the scatter plot of the data and the regression line
    axs[i].scatter(expenses, sales, color='blue', label='Training Data')
    axs[i].plot(expenses, regression_model.predict(expenses), color='red', linewidth=2, label='Linear Regression')
    axs[i].set_title(f'Linear Regression for {expense_type} Expenses')
    axs[i].set_xlabel(f'{expense_type} Expenses')
    axs[i].set_ylabel('Sales')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
