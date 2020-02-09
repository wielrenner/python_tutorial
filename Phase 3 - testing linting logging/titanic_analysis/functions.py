def extract_title(df):
    """
    Extracts the title out of a name

    Arguments:
        df {Pandas dataframe} -- should contain a Name column

    Returns:
        df {Pandas dataframe} -- same dataframe except changed Name column content
    """
    extraction = {'.*Mrs\..*': 'Mrs',
                  '.*Sir\..*': 'Royalty',
                  '.*Mr\..*': 'Mr',
                  '.*Capt\..*': 'Officer',
                  '.*Col\..*': 'Officer',
                  '.*Countess\..*': 'Royalty',
                  '.*Dona\..*': 'Royalty',
                  '.*Don\..*': 'Royalty',
                  '.*Dr\..*': 'Officer',
                  '.*Jonkheer.*': 'Royalty',
                  '.*Lady\..*': 'Royalty',
                  '.*Major\..*': 'Officer',
                  '.*Master\..*': 'Master',
                  '.*Mlle\..*': 'Miss',
                  '.*Mme\..*': 'Mrs',
                  '.*Ms\..*': 'Mrs',
                  '.*Rev\..*': 'Officer',
                  '.*Miss\..*': 'Miss'}
    df['Name'] = df['Name'].replace(extraction, regex=True)
    return df


def fillna_age(df):
    """
    Will fill all missing values of the Age column
    based on the median values of the Age
    after a groupby on the Name, Pclass, and Sex

    Arguments:
        df {Pandas dataframe} -- should contain a Age, Pclass, Name, and Sex column

    Returns:
        df {Pandas dataframe} -- same dataframe except all missing values of the Age column are filled
    """
    age_selection = df[['Age', 'Pclass', 'Name', 'Sex']].dropna()
    grouped_age = age_selection.groupby(['Name', 'Pclass', 'Sex'])['Age'].median()

    df['Age'] = df.apply(lambda x: grouped_age.loc[(x['Name'], x['Pclass'], x['Sex'])] if not x['Age'] > 0 else x['Age'],
                         axis=1)
    return df
