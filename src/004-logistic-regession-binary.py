'''
I started with 
https://towardsdatascience.com/logistic-regression-for-malignancy-prediction-in-cancer-27b1a1960184

But quickly realized that it was too advanced for what I was going for. 

I found a similar dataset in scikit learn and used that instead:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

'''

import numpy as np
from sklearn import datasets, linear_model

data = datasets.load_breast_cancer()
X = data.data.transpose()
y = data.target

# I'll learn feature selection at a later time.
# At that time, we'll see how we can
# 1) combine rid of highly correlated (redundant) features
# 2) identify which features have a logistical relationship with the target
# But for now, let us assume that part is done...
feature_indices = [0, 1, 4, 5, 6, 8]
feature_names = np.array(data.feature_names)[feature_indices]
X = X[feature_indices, :]

test_count = 20
X_train = X[:, :-test_count].transpose()
X_test = X[:, -test_count:].transpose()
y_train = y[:-test_count]
y_test = y[-test_count:]

# Train
regr = linear_model.LogisticRegression()
regr.fit(X_train, y_train)
print(
    'target = ' +
    ' + '.join(
        [
            f'{regr.coef_[0][i]:.2f} * {feature_names[i]}'
            for i in range(len(feature_names))
        ] + [
            f'{regr.intercept_[0]:.2f}'
        ]
    )
)


# Evaluate
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
print(f'accuracy = {score:.2f}')
