import pandas as pd
import numpy as np

from titanic_analysis.functions import extract_deck, calc_family_size, calc_name_length, fillna_embarked, label_encode


def test_extract_deck():
    df = pd.DataFrame({'cabin': ['A1', 'A2', 'B3', 'C4', np.nan]})
    df = extract_deck(df)

    assert 'deck' in df.columns, 'deck should be in the columns'
    assert 'A' in df['deck'].values, 'A should be in the decks'
    assert 'B' in df['deck'].values, 'B should be in the decks'
    assert 'C' in df['deck'].values, 'C should be in the decks'
    assert 'Z' in df['deck'].values, 'Z should be in the decks'
