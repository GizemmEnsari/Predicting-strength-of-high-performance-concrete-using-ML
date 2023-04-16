import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import Q2


# Load the training and test data
test_path = "test.csv"
train_path = "train.csv"

abs_path_test = os.path.abspath(test_path)
abs_path_train = os.path.abspath(train_path)

train_data = pd.read_csv(abs_path_train)
test_data = pd.read_csv(abs_path_test)

# Split the training data into features (X) and target variable (y)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# Define the Lasso regression model
lasso = Lasso()

# Set up a grid of hyperparameter values to search over
param_grid = {'alpha': np.logspace(-4, 4, 9)}

# Use cross-validation to search for the best hyperparameters
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best alpha using Lasso regression: {grid_search.best_params_['alpha']:.5f}")

# Train the final Lasso regression model with the best hyperparameters on the entire training dataset
lasso_final = Lasso(alpha=grid_search.best_params_['alpha']).fit(X_train, y_train)

# Evaluate the final model on the test dataset
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Compare the performance of the Lasso regression model with that of the simple linear regression model

reg = LinearRegression().fit(X_train, y_train)
Err_linear = np.mean((reg.predict(X_test) - y_test) ** 2)
print(f"Err using simple linear regression: {Err_linear:.5f}")


print(f"Err using ridge regression: {Q2.Err_test:.5f}")

Err_test = np.mean((lasso_final.predict(X_test) - y_test) ** 2)

print(f"Err using Lasso regression: {Err_test:.5f}")



# Plot the performance of the models explored during the alpha hyperparameter tuning phase as function of alpha
alphas = [x['alpha'] for x in grid_search.cv_results_['params']]
neg_mse = grid_search.cv_results_['mean_test_score']
plt.figure(figsize=(8,6))
plt.semilogx(alphas, neg_mse)
plt.xlabel('alpha')
plt.ylabel('Negative mean squared error')
plt.title('Lasso regression performance as a function of alpha')
plt.show()
