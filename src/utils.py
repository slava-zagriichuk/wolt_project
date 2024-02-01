# utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.font_manager import FontProperties

# Set up colors
colors = [
    "#001F3F", "#0074E4", "#3498DB",  # Blue Shades
    "#006400", "#228B22", "#00FF00",  # Green Shades
    "#8B0000", "#B22222", "#CD5C5C",  # Red Shades
    "#800080", "#9370DB", "#E6E6FA",  # Purple Shades
    "#FF8C00", "#FF6347", "#FA8072",  # Orange Shades
    "#404040", "#808080", "#D3D3D3"   # Gray Shades
]


def evaluate_errors(y_test, y_pred):
    """
    Evaluate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.

    Parameters:
    - y_test: true values
    - y_pred: predicted values

    Returns:
    - mse: Mean Squared Error
    - mae: Mean Absolute Error
    - r2: R-squared score
    """
    # Ensure that the inputs are numpy arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate R-squared (R2) score
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2


def show_evaluation(y_test, y_pred):
    """
    Plot true values, predicted values, and the absolute difference between them.

    Parameters:
    - y_true: true values
    - y_pred: predicted values
    """
    # Ensure that the inputs are numpy arrays
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate the absolute difference between true and predicted values
    diff = (y_pred - y_test)

    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Flatten the axes for easy numeration
    axes = axes.flatten()

    # Plot true values vs predicted values
    axes[0].plot(y_test, label='True values', color=colors[2], linewidth=1, alpha=0.7)
    axes[0].plot(y_pred, label='Predicted values', color=colors[7], linewidth=1, alpha=1)
    axes[0].set_title('True Values vs Predicted Values', color='lightgray')
    axes[0].set_ylabel('Couriers online', color='lightgray')
    axes[0].legend()

    # Plot the difference between true and predicted values
    axes[1].plot(diff, color=colors[8], label='Difference', linewidth=1, alpha=0.7)
    axes[1].axhline(0, linestyle='--', color=colors[11], label='Zero Difference Line', linewidth=1, alpha=0.7)
    axes[1].set_title('Difference between actual and predicted', color='lightgray')
    axes[1].set_ylabel('Difference', color='lightgray')
    axes[1].legend()

    for ax in axes:
        ax.set_facecolor('#222222')
        ax.grid(linewidth=0.5, color='lightgrey', alpha=0.2)
        ax.tick_params(colors='lightgrey')
    
    fig.set_facecolor('#222222')
    plt.xticks(fontproperties=FontProperties(family="Helvetica Neue", size=10))
    plt.yticks(fontproperties=FontProperties(family="Helvetica Neue", size=10))
    
    plt.suptitle('Model Evaluation', fontproperties=FontProperties(family="Helvetica Neue", size=16), color='lightgrey')
    plt.tight_layout()
    plt.show()
    
    
def define_dates_backwards(data, column, interval_months):
    # Find the minimum and maximum dates in the DataFrame
    max_date = data[column].max()
    min_date = data[column].min()
    
    # Setting the interval
    interval = pd.DateOffset(months=interval_months)
    
    # Setting the results list
    result = []
    
    # Append to the result all the dates backwards with given interval
    while max_date > min_date:
        result.append(str(max_date.date()))
        max_date = max_date - interval
        
    return result[1:]


def define_split_points(data, interval:int):
    length = len(data)
    current = length
    
    # Setting the results list
    result = []
    
    # Append to the result all the dates backwards with given interval
    while current > 0:
        result.append(current)
        current -= interval       
    else:
        result.append(0)
    
    tr_begin_list = result[2:]
    tr_fin = result[1]
    
    return tr_fin, tr_begin_list