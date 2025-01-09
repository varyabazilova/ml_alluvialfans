
# functions for the ML paper 
import pandas as pd
import os
import numpy as np
np.set_printoptions(precision=4)

import catboost
from catboost import *
from catboost import CatBoostClassifier, Pool, metrics, cv
from catboost.utils import get_roc_curve, get_confusion_matrix

import matplotlib.pyplot as plt

# other things: from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, recall_score, f1_score, confusion_matrix, precision_score
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import shap 


#  -------- functions to do things -----------

from sklearn.model_selection import StratifiedShuffleSplit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score

def strat_split_data_make_model_save_metrics(X, y, cat_features, seed, n_splits, model_params):
    ''' use this if you want to preserve the ratio of the class 
        when splitting the data to the train and test parts
        X = x features (df)
        y = target (df)
        cat_features = list['your', 'categorical', 'features'])
        seed = to keep reproductivity
        n_splits = how many times (1 for the model, more for CV) 
        model_params = parameters of the model''' 
    
    # Use StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
    
    # Create lists to store results
    models = []
    auc_scores = []
    accuracy_scores = []
    confusions = []

    
    # Split the data and create pools
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create CatBoost Pools
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
        
        # Create and fit the model
        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=test_pool, plot=False)
        
        # Store the trained model
        models.append(model)
        
        # Predict probabilities for the test set
        y_pred_proba = model.predict_proba(test_pool)[:, 1]  # Probability of the positive class
        
        # Calculate AUC score for each split
        auc = roc_auc_score(test_pool.get_label(), y_pred_proba)
        auc_scores.append(auc)
        
        # Calculate accuracy score for each split
        accuracy = accuracy_score(y_test, model.predict(X_test))
        accuracy_scores.append(accuracy)
        
        if n_splits == 1:
            # # confusion matrix
            conf_matrix = confusion_matrix(y_test, model.predict(X_test))
            confusions.append(conf_matrix)    
        
    return models, auc_scores, accuracy_scores, confusions
      
        

def split_data_make_model_save_metrics(X, y, cat_features, seed, n_splits, model_params):
    ''' use this if you dint care about preserving ratio of classes in the train/test split
        - split the data
        - fit the model
        - save metric
        - save model ''' 
        
    sss = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
    
    # Create lists to store results
    models = []
    auc_scores = []
    accuracy_scores = []
    
    # Split the data and create pools
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create CatBoost Pools
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
        
        # Create and fit the model
        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=test_pool, plot=False)
        
        # Store the trained model
        models.append(model)
        
        # Predict probabilities for the test set
        y_pred_proba = model.predict_proba(test_pool)[:, 1]  # Probability of the positive class
        
        # Calculate AUC score for each split
        auc = roc_auc_score(test_pool.get_label(), y_pred_proba)
        auc_scores.append(auc)
        
        # Calculate accuracy score for each split
        accuracy = accuracy_score(y_test, model.predict(X_test))
        accuracy_scores.append(accuracy)
    
    return models, np.array(auc_scores), accuracy_scores


# data quality assesent 
def invert_random_percentage(df, column_name, percentage, seed=None):
    """
    Randomly invert a percentage of values in a specified column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to invert values in.
    percentage (float): The percentage of values to invert (between 0 and 100).
    seed (int, optional): Random seed for reproducibility.

    Returns:
    pd.DataFrame: A copy of the DataFrame with inverted values in the specified column.
    """
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original

    if seed is not None:
        np.random.seed(seed)  # For reproducibility

    # Determine the number of values to change
    num_to_change = int(len(df_copy) * (percentage / 100))

    # Randomly select indices to change
    indices_to_change = np.random.choice(df_copy.index, num_to_change, replace=False)

    # Invert the values at the selected indices
    df_copy.loc[indices_to_change, column_name] = df_copy.loc[indices_to_change, column_name].apply(lambda x: 1 if x == 0 else 0)

    return df_copy







# ---------- output processing 
# Define function to categorize regions
def categorize_region(row):
    ''' Function to create region categories ''' 
    south = ['Central Himalaya', 'Eastern Himalaya', 'Western Himalaya']
    interior = ['Tibetan Interior Mountains', 'Altun Shan', 'Eastern Kunlun Shan', 'Western Kunlun Shan', 'Gangdise Mountains']
    west = ['Pamir Alay', 'Eastern Hindu Kush', 'Western Pamir', 'Eastern Pamir', 'Karakoram']
    east = ['Tanggula Shan', 'Nyainqentanglha', 'Hengduan Shan', 'Qilian Shan']
    # north: ['Eastern Tien Shan', 'Central Tien Shan', 'Dzhungarsky Alatau', 'Northern/Western Tien Shan']
    
    if row['region'] in south:
        return 'South'
    elif row['region'] in interior:
        return 'Interior'
    elif row['region'] in west:
        return 'West'
    elif row['region'] in east:
        return 'East'
    else:
        return 'North'

    
def reorder_columns_for_colors(df):
    order = [
        'elv_median', 'elv_range', 'area_m2', 'perim_m', 'sl_median','M',
       'circularity_ratio', 'compactness', 
         
       'tp', 'max_annualsum_tp', 'n_rainydays_median', 'precip95', 'snow', 'rain',
       'mean_annual_t2m_downsc', 'cross_zero', 'frost_days', 
         
       'veg_frac', 'cont_permafrost', 'glacier']
    df = df.reindex(columns=order)
    return df

def rename_columns_for_colors(df):
    df_names = df.rename(columns={'elv_median':'(1) median elevation',
                                  'elv_range': '(2) relief', 
                                  'area_m2': '(3) area',
                                  'perim_m': '(4) perimeter', 
                                  'sl_median': '(5) median slope',
                                  'M': '(6) Melton ratio',
                                  'circularity_ratio':'(7) circularity ratio', 
                                  'compactness':'(8) compactness', 
                                          
                                  'tp': '(9) total annual precipitation', 
                                  'max_annualsum_tp': '(10) max annual precipitation', 
                                  'n_rainydays_median': '(11) N of wet days',
                                  'precip95': '(12) 95% precipitation',
                                  'snow': '(13) snowfall', 
                                  'rain': '(14) rainfall',
                                  'mean_annual_t2m_downsc': '(15) mean annual temperature',
                                  'cross_zero':'(16) thermal weathering',
                                  'frost_days': '(17) frost weathering', 
      
                                  'veg_frac': '(18) vegetation cover (%)',
                                  'cont_permafrost': '(19) continious permafrost', 
                                  'glacier':'(20) glacier'})
    return df_names




def rename_columns_morph(df):
    df_names = df.rename(columns={'elv_median':'median elevation',
                                  'elv_range': 'relief', 
                                  'area_m2': 'area',
                                  'perim_m': 'perimeter', 
                                  'sl_median': 'median slope',
                                  'M': 'Melton ratio',
                                  'circularity_ratio':'circularity ratio', 
                                  'compactness':'compactness'})
    return df_names

def rename_columns_clim(df):
    df_names = df.rename(columns={'tp': 'total annual precipitation', 
                                  'max_annualsum_tp': 'max annual precipitation', 
                                  'n_rainydays_median': 'N of wet days',
                                  'precip95': '95% precipitation',
                                  'snow': 'snowfall', 
                                  'rain': 'rainfall',
                                  'mean_annual_t2m_downsc': 'mean annual temperature',
                                  'cross_zero':'thermal weathering',
                                  'frost_days': 'frost weathering'})
    return df_names




def determine_class(value):
    if value == 'TP': #TP
        return 1
    elif value ==  'TN': #TN
        return 1
    elif value == 'FP': #FP
        return 0
    elif value == 'FN': #FN
        return 0 


def determine_probability_bin(value):
    if value >= 0.0 and value <= 0.05:
        return '0-5'
    if value > 0.05 and value <= 0.10:
        return '5-10'
    elif value > 0.10 and value <= 0.15:
        return '10-15'
    elif value > 0.15 and value <= 0.20:
        return '15-20'
    elif value > 0.20 and value <= 0.25:
        return '20-25'
    elif value > 0.25 and value <= 0.30:
        return '25-30'
    elif value > 0.30 and value <= 0.35:
        return '30-35'
    elif value > 0.35 and value <= 0.40:
        return '35-40'
    elif value > 0.40 and value <= 0.45:
        return '40-45'    
    elif value > 0.45 and value <= 0.50:
        return '45-50'
    elif value > 0.50 and value <= 0.55:
        return '50-55'
    elif value > 0.55 and value <= 0.60:
        return '55-60'
    elif value > 0.60 and value <= 0.65:
        return '60-65'
    elif value > 0.65 and value <= 0.70:
        return '65-70'
    elif value > 0.70 and value <= 0.75:
        return '70-75'
    elif value > 0.75 and value <= 0.80:
        return '75-80'
    elif value > 0.80 and value <= 0.85:
        return '80-85'
    elif value > 0.85 and value <= 0.90:
        return '85-90'
    elif value > 0.90 and value <= 0.95:
        return '90-95'
    elif value > 0.95 and value <= 1.0:
        return '95-100'
    
def determine_order_for_bins_plot(value):
    if value >= 0.0 and value <= 0.05:
        return '0'
    if value > 0.05 and value <= 0.10:
        return '1'
    elif value > 0.10 and value <= 0.15:
        return '10'
    elif value > 0.15 and value <= 0.20:
        return '15'
    elif value > 0.20 and value <= 0.25:
        return '20'
    elif value > 0.25 and value <= 0.30:
        return '25'
    elif value > 0.30 and value <= 0.35:
        return '30'
    elif value > 0.35 and value <= 0.40:
        return '35'
    elif value > 0.40 and value <= 0.45:
        return '40'    
    elif value > 0.45 and value <= 0.50:
        return '45'
    elif value > 0.50 and value <= 0.55:
        return '50'
    elif value > 0.55 and value <= 0.60:
        return '55'
    elif value > 0.60 and value <= 0.65:
        return '60'
    elif value > 0.65 and value <= 0.70:
        return '65'
    elif value > 0.70 and value <= 0.75:
        return '70'
    elif value > 0.75 and value <= 0.80:
        return '75'
    elif value > 0.80 and value <= 0.85:
        return '80'
    elif value > 0.85 and value <= 0.90:
        return '85'
    elif value > 0.90 and value <= 0.95:
        return '90'
    elif value > 0.95 :
        return '95'
    
