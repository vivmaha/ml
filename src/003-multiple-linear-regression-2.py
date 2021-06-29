'''

Taking my learnings from ./002-multiple-linear-regression.py, and applying it to
the dataset from ./001-linear-regression.py

'''

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

data = datasets.load_diabetes()
X = data.data.transpose()
y = data.target


def check_for_linear_relationships():
    '''
    Plots scatter plots of all columns vs. stock index price to visually 
    determine which columns have a linear relationship with it. 
    '''
    nrows = 1
    ncols = len(X)
    plt.title('test')
    for i in range(0, ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.scatter(X[i], y)
        plt.xlabel(data.feature_names[i])
    plt.suptitle(
        "Scatter plots of each feature (x) vs. disease progression (y)")
    plt.show()


# check_for_linear_relationships()
'''
The above visualization shows that "bmi", "s5" and "s6" are linearly related
with disease progression. We'll include these features.
'''

feature_indices = [2, 8, 9]
feature_names = np.array(data.feature_names)[feature_indices]
X = X[feature_indices, :]

test_count = 20
X_train = X[:, :-test_count].transpose()
X_test = X[:, -test_count:].transpose()
y_train = y[:-test_count]
y_test = y[-test_count:]

# Train
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(
    'target = ' +
    ' + '.join(
        [
            f'{regr.coef_[i]:.2f} * {feature_names[i]}'
            for i in range(len(feature_names))
        ] + [
            f'{regr.intercept_:.2f}'
        ]
    )
)

# Evaluate
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
print(f'r2 = {score:.2f}')
