import datetime as dt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def timeit(f):
    def wrapper(df, *args, **kwargs):
        tic = dt.datetime.now()
        result = f(df, *args, **kwargs)
        toc = dt.datetime.now()
        print(f'{f.__name__} took {toc-tic}')
        return result
    return wrapper

@timeit
def extract_deck(df):
    df['deck'] = df['cabin'].str[0]
    df['deck'].fillna('Z', inplace=True)
    return df

@timeit
def calc_family_size(df):
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    
    bins = [0, 1, 4, 100]
    group_names = ['singleton', 'small', 'large']
    df['family_size_cat'] = pd.cut(df['family_size'], bins, labels=group_names)
    return df

@timeit
def calc_name_length(df):
    df['name_length'] = df['name'].apply(lambda x: len(x))
    
    bins = [0, 20, 40, 57, 85]
    group_names = ['short', 'ok', 'good', 'long']
    df['name_length_cat'] = pd.cut(df['name_length'], bins, labels=group_names)
    return df

@timeit
def fillna_embarked(df):
    df['embarked'].fillna('S', inplace=True)
    return df

@timeit
def label_encode(df):
    labelEnc = LabelEncoder()

    cat_vars = ['embarked', 'sex', 'family_size_cat', 'name_length_cat', 'deck']
    for col in cat_vars:
        df[col] = labelEnc.fit_transform(df[col])
    return df