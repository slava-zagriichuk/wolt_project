# data_preprocessing.py

import pandas as pd
import numpy as np


def handle_missing_values(data: pd.Series, method='linear'):
    """
    Handle missing values in a pandas DataFrame using interpolation.

    Parameters:
    - data (pd.Series): The input DataFrame.
    - method (str): The interpolation method. Default is 'linear'. Other options: 'polynomial', 'spline', etc.

    Returns:
    - pd.DataFrame: A DataFrame with missing values handled using interpolation.
    """
    try:
        # Using interpolate method with the specified interpolation method
        data = data.interpolate(method=method)

        print(f"Missing values handled using interpolation method: {method}")
        return data
    except Exception as e:
        print(f"Error handling missing values: {str(e)}")
        return None


def handle_outliers(data, column, multiplier=1.5, replacement_strategy='nan'):
    """
    Handle outliers in a specific column of a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - column (str): The column containing the values to handle outliers.
    - multiplier (float): The multiplier to determine the outlier threshold. Default is 1.5.
    - save (bool): Save dataframe with outliers to the file anomalies.csv for further study
    - replacement_strategy (str): The strategy to replace outliers. Options: 'median', 'nan', 'interpolate' or a specific value.

    Returns:
    - pd.Series: A series with outliers handled in the specified column.
    """
    try:
        # Calculate Q1, Q3, and IQR
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1

        # Define the outlier threshold
        threshold = q3 + multiplier * iqr

        # Replace values beyond the threshold with the specified strategy
        if replacement_strategy == 'median':
            replacement_value = data[column].median()
        elif replacement_strategy == 'nan' or replacement_strategy == 'interpolate':
            replacement_value = np.nan
        else:
            replacement_value = replacement_strategy
        
        data[column] = np.where(data[column] > threshold, replacement_value, data[column])
        
        if replacement_strategy == 'interpolate':
            data[column] = data[column].interpolate()

        print(f"Outliers handled using threshold: {threshold}")
        return data[column]
    except Exception as e:
        print(f"Error handling outliers: {str(e)}")
        return None

def show_outliers(data, column, multiplier=1.5):
    """
    Handle outliers in a specific column of a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - column (str): The column containing the values to handle outliers.
    - multiplier (float): The multiplier to determine the outlier threshold. Default is 1.5.
    
    Returns:
    - pd.Dataframe: Dataframe with outliers to study them
    """
    try:
        # Calculate Q1, Q3, and IQR
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1

        # Define the outlier threshold
        threshold = q3 + multiplier * iqr
        
        result = data[data[column] > threshold]
        if not result.empty:
            print(f"There are outliers in column: {column} over threshold {threshold}")
        return result
    except Exception as e:
        print(f"Error spotting outliers: {str(e)}")
        return None


def parse_dates(data: pd.Series):
    """
    Parse date strings in a pandas Series into datetime objects.

    Parameters:
    - series (pd.Series): The input Series containing date strings.

    Returns:
    - pd.Series: A new Series with date strings parsed into datetime objects.
    """
    try:
        parsed_series = pd.to_datetime(data)

        print("Dates parsed successfully")
        return parsed_series
    except Exception as e:
        print(f"Error parsing dates: {str(e)}")
        return None


def extract_day_names(data: pd.Series):
    """
    Extract day names from a pandas Series containing datetime objects.

    Parameters:
    - data (pd.Series): The input Series containing datetime objects.

    Returns:
    - pd.Series: A new Series with day names extracted.
    """
    try:
        # Use the dt accessor to access datetime properties, and day_name() to get day names
        day_names_series = data.dt.day_name()

        print("Day names extracted successfully from the Series")
        return day_names_series
    except Exception as e:
        print(f"Error extracting day names: {str(e)}")
        return None
    
def extract_day_category(data: pd.Series):
    """
    Extract day categories from a pandas Series containing datetime objects.

    Parameters:
    - data (pd.Series): The input Series containing datetime objects.

    Returns:
    - pd.Series: A new Series with day category extracted.
    """
    categories = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 2,
        'Thursday': 1,
        'Friday': 0,
        'Saturday': 0,
        'Sunday': 1
    }
    
    try:
        # Use the dt accessor to access datetime properties, and day_name() to get day names
        day_category_series = data.apply(lambda d: categories[d.day_name()])

        print("Day categories extracted successfully from the Series")
        return day_category_series
    except Exception as e:
        print(f"Error extracting day names: {str(e)}")
        return None 


def extract_days_from_beginning(data: pd.Series):
    """
    Creates a new Series representing the number of days from the beginning
    for each date in the input date series.

    Parameters:
    - date_series: pandas Series (datetime format)
        The Series containing dates for which the 'days_from_beginning' values
        will be calculated.

    Returns:
    - pandas Series
        A new Series representing the number of days from the beginning.
    """
    # Calculate the 'days_from_beginning' values
    days_from_beginning = (data - data.min()).dt.days

    return days_from_beginning