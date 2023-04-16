# Predicting-strength-of-high-performance-concrete-using-ML
Predicting strength of high-performance concrete using ML algorithms:

### In the Q1.py file:
I trained a multivariate ordinary least squares ("simple") linear regression model to predict the compressive strength of an input concrete mixture based on the relevant features. I used only the "training" dataset and estimated the "Err" using both the validation approach and a cross-validation (CV) approach.

To estimate the "Err" using the validation approach, I split the "training" dataset into "train/validation/test" subsets. I trained the model on the "train + validation" subset and tested it on the "test" subset. I then calculated the mean squared error (MSE) between the predicted and actual values of the compressive strength to estimate the "Err".

For the CV approach, I used a k-fold cross-validation, where the dataset was split into k equal parts (or "folds"). The model was trained k times, each time using a different fold as the validation set and the remaining folds as the training set. The mean of the k MSE values was then calculated to estimate the "Err".

I carefully considered the choice of the number of folds used in the CV approach. A higher number of folds resulted in a more accurate estimate of the "Err", but also increased the computational time required. Conversely, a lower number of folds reduced the computational time required but may have resulted in a less accurate estimate of the "Err".

Finally, I compared the "Err" estimates obtained using the validation and CV approaches.

### In the Q2.py file:

I trained a multivariate Ridge regression model for the concrete compressive strength prediction task. I used the "training" dataset and a CV based grid-search approach to tune the regularization parameter "α" of the Ridge regression model. Using the "best" "α" setting, I re-trained the model on the entire "training" dataset to obtain the final Ridge regression model. I estimated the "Err" of this final model on the "test" dataset.

To plot the performance of the models explored during the "α" hyperparameter tuning phase as a function of "α", I first trained Ridge regression models with different values of "α" using the CV approach. I then calculated the mean squared error (MSE) for each model and plotted the MSE values as a function of "α".

To compare the performance of the Ridge regression model with that of the "simple" linear regression model, I compared the "Err" estimates obtained from both models on the "test" dataset. I also compared the MSE values of the models during the "α" hyperparameter tuning phase.

Overall, the Ridge regression model performed better than the "simple" linear regression model. The Ridge regression model had a lower "Err" estimate on the "test" dataset and had lower MSE values during the "α" hyperparameter tuning phase. The Ridge regression model was able to better handle multicollinearity in the dataset, leading to improved performance compared to the "simple" linear regression model.


### In the Q3.py file:

I repeated the above experiment with a multivariate Lasso regression model. Similar to the Ridge regression model, I used the "training" dataset and a CV based grid-search approach to tune the regularization parameter "α" of the Lasso regression model. Using the "best" "α" setting, I re-trained the model on the entire "training" dataset to obtain the final Lasso regression model. I estimated the "Err" of this final model on the "test" dataset.

To plot the performance of the models explored during the "α" hyperparameter tuning phase as a function of "α", I trained Lasso regression models with different values of "α" using the CV approach. I then calculated the mean squared error (MSE) for each model and plotted the MSE values as a function of "α".

To compare the performance of the final Lasso regression model with that of both the Ridge regression and the "simple" linear regression models, I compared the "Err" estimates obtained from all three models on the "test" dataset. I also compared the MSE values of the models during the "α" hyperparameter tuning phase.

The performance of the Lasso regression model was generally similar to that of the Ridge regression model, and both models performed better than the "simple" linear regression model. However, the Lasso regression model was able to perform feature selection by shrinking the coefficients of irrelevant features to zero. This resulted in a simpler model and improved interpretability.

When comparing the MSE values during the "α" hyperparameter tuning phase, the Lasso regression model tended to perform better than the Ridge regression model for smaller values of "α". However, for larger values of "α", the Ridge regression model performed better.

Overall, the choice between the Lasso and Ridge regression models depends on the specific needs of the problem at hand, including the importance of feature selection and the desired level of model complexity.
