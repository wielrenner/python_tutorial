import pandas as pd
import numpy as np

from titanic_analysis.functions import fillna_age


def test_fillna_age():
    df = pd.DataFrame({'Age': [1, 2, 3, 4, np.nan, np.nan],
                       'Pclass': ['A', 'A', 'B', 'B', 'A', 'B'],
                       'Name': ['Mr', 'Mr', 'Mrs', 'Mrs', 'Mr', 'Mrs'],
                       'Sex': ['M', 'M', 'F', 'F', 'M', 'F']})
    df = fillna_age(df)

    assert 'Age' in df.columns, 'Age should be in the columns'
    assert df.loc[4, 'Age'] == 1.5, 'Age should be 1.5'
    assert df.loc[5, 'Age'] == 3.5, 'Age should be 3.5'
