import pandas as pd
__author__ = "Nicholas Marsano, Stefanos Stoikos, Simon Heck"
"""
This class contains different utility fucntions that clean and scale the data for use in the classifiers
"""
def remove_NA_rows(df):
    columns = df.keys()
    for column in columns:
        df = df[df[column].notna()]
    return df

def mean_normalization(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df

def min_max_normalization(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def clean_data(data_path,label_column,normalization=0):
    #if normalization 0 then min max else mean
    df = pd.read_csv(data_path)
    df = remove_NA_rows(df)
    lblC = df[label_column].tolist()
    df = df.drop([label_column], axis=1)
    if(normalization):
        df = min_max_normalization(df)
    else:
        df = mean_normalization(df)
    df.insert(loc=0,column=label_column,value=lblC)
    return df