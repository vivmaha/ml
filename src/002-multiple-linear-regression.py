'''

I followed along the code from here:
https://datatofish.com/multiple-linear-regression-python/

...but made it my own to understand it.

After learning to visually determine which features to learn, I decided to
switch to another dataset. See the comments at the end for more details.

'''

# higher-level array manipulations than numpy
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

Stock_Market = {
    'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
    'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1, 6.1, 5.9, 6.2, 6.2, 6.1],
    'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958, 971, 949, 884, 866, 876, 822, 704, 719]
}

df = pd.DataFrame(Stock_Market)


def check_for_linear_relationships():
    '''
    Plots scatter plots of all columns vs. stock index price to visually 
    determine which columns have a linear relationship with it. 
    '''
    nrows = 1
    ncols = len(Stock_Market.keys())
    for i in range(0, ncols):
        x_column = list(Stock_Market.keys())[i]
        y_column = 'Stock_Index_Price'
        plt.subplot(nrows, ncols, i + 1)
        plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
    plt.show()


# check_for_linear_relationships()
'''
 ^ This showed that both Interest_Rate and Unemployment_Rate are linearly
   related to Stock_Index_Price, and therefore would be good candidates to
   include in a linear regression.
'''

X = df[['Interest_Rate', 'Unemployment_Rate']]
Y = df['Stock_Index_Price']
regr = linear_model.LinearRegression()
regr.fit(X, Y)

'''
At this stage I'm going to proceed with the scikit learn dataset from 
001-linear-regression.py. See 003-multiple-linear-regression-2.py for the 
rest.
'''
