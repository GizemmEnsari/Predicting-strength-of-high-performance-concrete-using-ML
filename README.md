# Predicting-strength-of-high-performance-concrete-using-ML
Predicting strength of high-performance concrete using ML algorithms:

###In the Q1.py file:
I trained a multivariate ordinary least squares ("simple") linear regression model to predict the compressive strength of an input concrete mixture based on the relevant features. I used only the "training" dataset and estimated the "Err" using both the validation approach and a cross-validation (CV) approach.

To estimate the "Err" using the validation approach, I split the "training" dataset into "train/validation/test" subsets. I trained the model on the "train + validation" subset and tested it on the "test" subset. I then calculated the mean squared error (MSE) between the predicted and actual values of the compressive strength to estimate the "Err".

For the CV approach, I used a k-fold cross-validation, where the dataset was split into k equal parts (or "folds"). The model was trained k times, each time using a different fold as the validation set and the remaining folds as the training set. The mean of the k MSE values was then calculated to estimate the "Err".

I carefully considered the choice of the number of folds used in the CV approach. A higher number of folds resulted in a more accurate estimate of the "Err", but also increased the computational time required. Conversely, a lower number of folds reduced the computational time required but may have resulted in a less accurate estimate of the "Err".

Finally, I compared the "Err" estimates obtained using the validation and CV approaches.

###In the Q2.py file:
