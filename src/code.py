import numpy as np
import pandas as pd

# Function to separate the dataframe by data type

def separate_df(dataframe):
    num_df = data.select_dtypes(include=['number'])
    cat_df = data.select_dtypes(exclude=['number'])
    return num_df, cat_df