import numpy as np
import pandas as pd

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    return df

# Function to separate the dataframe by data type
def separate_df(dataframe):
    num_df = data.select_dtypes(include=['number'])
    cat_df = data.select_dtypes(exclude=['number'])
    return num_df, cat_df

# plotting_missingness
def plot_missingness(data):
    plt.figure(figsize=(15, 3))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
    plt.title("Missingness Visualization")
    plt.show()
    
def plot_boxplots(data):
    num_cols = len(data.columns)
    num_rows = (num_cols + 2) // 3  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))  # Create subplots

    for i, column in enumerate(data.columns):
        ax = axes[i // 3, i % 3] if num_rows > 1 else axes[i % 3]
        sns.boxplot(x=data[column], ax=ax)
        sns.stripplot(x=data[column], color='magenta', size=1, ax=ax)
        ax.set_title(f'Boxplot of {column}')  # Set the title

    plt.tight_layout()  
    plt.show()  
    

def plot_kdeplots(data):
    num_cols = len(data.columns)
    num_rows = (num_cols + 2) // 3  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))  # Create subplots

    for i, column in enumerate(data.columns):
        ax = axes[i // 3, i % 3] if num_rows > 1 else axes[i % 3]  # Select subplot
        sns.histplot(data[column], ax=ax,kde=True)  # Plot boxplot for the column
        ax.set_title(f'Distribution of {column}')  # Set the title

    plt.tight_layout()  # Adjust layout
    plt.show()  # Show the plots
    
    
def plot_boxplots_numerical_vs_target(data, target_column='booking_complete', num_cols=3, figsize=(30, 25)):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_rows = (len(numerical_columns) + num_cols - 1) // num_cols

    plt.figure(figsize=figsize)
    for i, num_col in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(x=target_column, y=num_col, data=data)
        plt.title(f'{num_col} vs. {target_column}')
        plt.xlabel(target_column)
        plt.ylabel(num_col)

    plt.tight_layout()
    plt.show()
    

def plot_countplots_categorical_vs_target(data, target_column='booking_complete', num_cols=3, figsize=(30, 25)):
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_rows_cat = (len(categorical_columns) + num_cols - 1) // num_cols

    plt.figure(figsize=figsize)
    for i, cat_col in enumerate(categorical_columns, 1):
        plt.subplot(num_rows_cat, num_cols, i)
        sns.countplot(x=cat_col, hue=target_column, data=data)
        plt.title(f'{cat_col} vs. {target_column}')
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        plt.legend(title=target_column, loc='upper right')

    plt.tight_layout()
    plt.show()
