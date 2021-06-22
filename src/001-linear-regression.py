'''

I followed along the code from here:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

...but made it my own to understand it.

'''

# scikit-learn is the ml library
from sklearn import datasets, linear_model

# NumPy used for multi-dimensional array manipulation
# datasets from scikit-learn are NumPy structures
import numpy as np

# matplotlib used to plot data
import matplotlib.pyplot as plt

diabetes_X_all_features, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X_all_features[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_y_train = diabetes_y[:-20]

diabetes_X_test = diabetes_X[-20:]
diabetes_y_test = diabetes_y[-20:]

# Train
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
m = regr.coef_[0]
c = regr.intercept_
print(f'y = {m:.2f}x + {c:.2f}')

# Evaluate
diabetes_y_pred = regr.predict(diabetes_X_test)
score = regr.score(diabetes_X_test, diabetes_y_test)
print(f'r2 = {score:.2f}')

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.show()
