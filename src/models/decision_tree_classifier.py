#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def decision_tree_classifier(dataframe, currencies=None, include_sp500=True,lag=1,train_size=0.75, seed=42,long_only=False,max_depth=10):
    """
    Trains a Decision Tree Classifier on financial data to predict binary outcomes and evaluates its performance.

    This function processes financial data, which can include currency pairs and optionally S&P 500 data, and applies a Decision Tree Classifier for binary classification. It calculates the model's accuracy and plots cumulative returns for comparison.

    Parameters:
    - dataframe (pd.DataFrame): The dataset containing the financial data. It should have a 'DATE' column, currency data columns, and an 'SP500' column.
    - currencies (list of str, optional): List of currency columns to include in the analysis. If None, all available currencies in the dataframe will be used.
    - include_sp500 (bool): Determines whether to include the S&P 500 data in the analysis. Defaults to True.
    - lag (int): The number of periods by which to lag the response variable for prediction. Defaults to 1.
    - train_size (float): The proportion of the dataset to use for training the model. The rest will be used for testing. Defaults to 0.75.
    - seed: The seed can be set manually such that the results are reproducible. Default is 42.
    - max_depth (int): The maximum depth of the decision tree. Helps to control the complexity of the model. Defaults to 10.

    Returns:
    - tuple: A tuple containing a DataFrame of model accuracies and another DataFrame with the cumulative returns for both the 'Long' strategy and the Decision Tree Classifier model.
    """
    # Setting up response and regressor variables
    y1 = dataframe['SP500']  # set up response variable
    # Calculating binary response
    y1_binary = (y1 > 0).astype(int)  # For positive Return 1 for negative Return 0
    y_dates = dataframe['DATE']

    # Selecting columns based on currencies if provided.
    if currencies is not None:
        unavailable_currencies = [currency for currency in currencies if currency not in dataframe.columns]
        if unavailable_currencies:
            available_currencies = ', '.join([col for col in dataframe.columns if col not in ['DATE', 'SP500']])
            unavailable_currencies_str = ', '.join(unavailable_currencies)
            print(f"Sorry, {unavailable_currencies_str} is not an available currency pair. Please choose currency pairs from: {available_currencies}")
            return None
        else:
            X1 = dataframe[currencies]  # X --> explanatory variables without target
    else:
        X1 = dataframe.drop(['DATE', 'SP500'], axis=1)
    X1 = X1.shift(lag).dropna()  # Shift to prevent look ahead bias

    # Including or excluding SP500 based on the flag
    if include_sp500:
        x1_spx = dataframe['SP500'].shift(lag).dropna() # select SP500 with a lag as additional explanatory variable
        X1 = pd.concat([X1, x1_spx], axis=1) # adding SP500 to explanatory variables

    y1 = y1.iloc[lag:]  # Remove first lag observations
    y1_binary = y1_binary.iloc[lag:]  # Remove first lag observations

    # get the returns from the log returns
    y1_exp = np.exp(y1) - 1

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X1, y1_binary, random_state=seed, shuffle=False, test_size=1-train_size)

    # Setting up dates
    y_test_dates = y_dates[y_test.index[0] - lag:]
    
    # Training the Decision Tree Classifier
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree_clf.fit(X_train, y_train)
    y_pred_train_dtc = tree_clf.predict(X_train)
    y_pred_test_dtc = tree_clf.predict(X_test)

    # Changing all 0 to -1 for return calculation if long_only is set to false
    if long_only==False:
        y_pred_test_dtc[np.where(y_pred_test_dtc == 0)] = -1

    # Calculating returns
    y1_ret = y1_exp
    y_bench = y1_ret[y_test.index[0]-lag:]
    y_long = np.cumsum(y_bench)
    y_dtc = np.cumsum(y_bench * y_pred_test_dtc)

    # Calculating accuracies
    accuracies = pd.DataFrame({
        "Classifiers": ["DTC"],
        "in sample": [accuracy_score(y_train, y_pred_train_dtc)],
        "out of sample": [accuracy_score(y_test, y_pred_test_dtc)]
    }).set_index('Classifiers')

    test_perform = pd.DataFrame({"SP500": y_long, "DTC": y_dtc})

    cumreturns = pd.concat([y_test_dates, test_perform], axis=1).dropna()

    return accuracies, cumreturns

# In[ ]:




