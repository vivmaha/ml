'''
EDA w/ pandas

Following along this tutorial
https://www.youtube.com/watch?v=9m4n2xVzk9o
'''

import pandas as pd
from pandas.core.frame import DataFrame
from io import StringIO


def scrape_data():
    url = 'https://www.basketball-reference.com/leagues/NBA_2019_per_game.html'
    df = pd.read_html(url)[0]
    return df


def reset_df_dtypes(df: DataFrame):
    # saving as CSV and re-reading it causes the dtypes to get re-inferred
    return pd.read_csv(StringIO(df.to_csv(index=False)))


def clean_data(df: DataFrame):
    # the headers are duplicated in a row every 20 rows or so. remove them.
    df = df.drop(df[df.Age == 'Age'].index)

    df = df.fillna(0)

    df = df.drop(['Rk'], axis=1)

    # now that we've cleaned it, we can get better types (eg float instead of
    # object)
    df = reset_df_dtypes(df)
    return df


df = scrape_data()
df = clean_data(df)


# I followed along by issuing commands in the interpreter here.
# Perhaps now would be a good time to pick up Jupyter notebooks..
