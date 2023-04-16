import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

test_path = "test.csv"
train_path = "train.csv"

abs_path_test = os.path.abspath(test_path)
abs_path_train = os.path.abspath(train_path)

train_data = pd.read_csv(abs_path_train)
test_data = pd.read_csv(abs_path_test)



# load the training data


# split the training data into features (X) and target variable (y)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# train a linear regression model using ordinary least squares
reg = LinearRegression().fit(X_train, y_train)

# load the test data


# split the test data into features (X) and target variable (y)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# validate the model using the validation approach
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

reg_val = LinearRegression().fit(X_train_val, y_train_val)
Err_val = np.mean((reg_val.predict(X_val) - y_val) ** 2)

reg_val_train = LinearRegression().fit(np.concatenate([X_train_val, X_val]), np.concatenate([y_train_val, y_val]))
Err_val_train = np.mean((reg_val_train.predict(X_train_val) - y_train_val) ** 2)

reg_test = LinearRegression().fit(np.concatenate([X_train_val, X_val]), np.concatenate([y_train_val, y_val]))
Err_test = np.mean((reg_test.predict(X_test) - y_test) ** 2)

print(f"Err using validation approach: {Err_test:.5f}")

# validate the model using the cross-validation approach
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(reg, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
Err_cv = -np.mean(scores)

print(f"Err using cross-validation approach: {Err_cv:.5f} (number of folds = 5)")

