def add_categorical_aggregations(df):

    # Define feature types
    categorical_cols = ['Sex']
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']

    # Single categorical column case (simplified from original loop)
    for cat_col in categorical_cols:
        # Calculate min/max aggregations for all numerical columns
        aggs = df.groupby(cat_col)[numerical_cols].agg(['min', 'max'])

        # Flatten multi-index columns
        aggs.columns = [f"{cat_col}_{num_col}_{stat}" 
                       for num_col, stat in aggs.columns]

        # Merge with original data
        df = df.merge(aggs, on=cat_col, how='left')

    return df

# Usage example:
train = add_categorical_aggregations(train)
test = add_categorical_aggregations(test)
