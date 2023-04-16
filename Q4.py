import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define the function to load the data from the given directory

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    data =[X_train,y_train,X_test]
    return data

# Define the function to predict the compressive strength using the random forest regressor
def predictCompressiveStrength(data_dir):
    # Load the training and test data
    X_train, y_train= load_data(data_dir)[0] , load_data(data_dir)[1]

    # Train a random forest regressor
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rfr.fit(X_train, y_train)

    # Predict the compressive strength on the test data
    y_pred = rfr.predict(load_data(data_dir)[2])
    return y_pred

# example running of the program

load_data("/Users/gizemmensari/Desktop/COURSES <3/comp 3202/assignment 2/")
print(predictCompressiveStrength("/Users/gizemmensari/Desktop/COURSES <3/comp 3202/assignment 2"))




