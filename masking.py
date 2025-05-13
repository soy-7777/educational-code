def add_interactions_onehot(df, features, gender_col='Sex'):

    # Create one-hot columns (no need for get_dummies - we know it's binary)
    df['Male'] = df[gender_col]  # 1 if male, 0 otherwise
    df['Female'] = 1 - df[gender_col]  # Inverse

    # Create interactions
    for feat in features:
        df[f'{feat}_x_Male'] = df[feat] * df['Male']
        df[f'{feat}_x_Female'] = df[feat] * df['Female']

    # Drop temporary one-hot columns (optional)
    df.drop(['Male', 'Female'], axis=1, inplace=True)

    return df

# Usage
train = add_interactions_onehot(train, features=['Duration', 'Heart_Rate', 'Body_Temp'])
test = add_interactions_onehot(test, features=['Duration', 'Heart_Rate', 'Body_Temp'])
