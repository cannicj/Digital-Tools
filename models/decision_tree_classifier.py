#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def decision_tree_classifier(dataframe, currencies=None, include_sp500=True,lag=1,train_size=0.75,max_depth=10):
     """
    Applies a Decision Tree Classifier to financial data for binary outcome prediction.

    Parameters:
    - DataFrame: The main dataset including currency and optionally S&P 500 data.
    - currencies (list of str, optional): List of currency pair columns from the dataset. Includes all if None.
    - train_size (float): Proportion of the dataset used for training the model. Default is 0.75.
    - include_sp500 (bool): Flag to decide if S&P 500 data should be included. Default is False.
    - lag (int): Number of periods to lag the S&P 500 data. Default is 1.
    - max_depth (int): Maximum depth of the decision tree. Default is 10.

    Returns:
    - tuple: Contains DataFrame of model accuracies and DataFrame or plot of model's financial performance.
    """
    
    # Selecting columns based on currencies if provided
    if currencies is not None:
        unavailable_currencies = [currency for currency in currencies if currency not in dataframe.columns]
        if unavailable_currencies:
            available_currencies = ', '.join([col for col in dataframe.columns if col not in ['DATE', 'SP500']])
            unavailable_currencies_str = ', '.join(unavailable_currencies)
            print(f"Sorry, {unavailable_currencies_str} is not an available currency pair. Please choose currency pairs from: {available_currencies}")
            return None
    
    # Setting up response and regressor variables
    y1 = dataframe.iloc[lag:, -1]  # Assuming the last column is the response variable
    y_dates = dataframe.iloc[lag:, 0]
    X1 = dataframe.iloc[:-lag, 1:-1]  # Excluding Date and response variable

    # Including or excluding SP500 based on the flag
    if include_sp500:
        x1_spx = dataframe.iloc[:-lag, -1]
        X1 = pd.concat([X1, x1_spx], axis=1)

    # Calculating binary response
    y1_exp = np.exp(y1) - 1
    y1_binary = (y1_exp > 0).astype(int)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X1, y1_binary, random_state=42, shuffle=False, test_size=1-train_size)

    # Setting up dates
    y_test_dates = y_dates[y_test.index[0]-lag:]
    
    # Training the Decision Tree Classifier
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_train_dtc = tree_clf.predict(X_train)
    y_pred_test_dtc = tree_clf.predict(X_test)
    

    
    # Changing all 0 to -1 for return calculation
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

    print(accuracies)

    # Plotting
    plt.figure(figsize=(14, 6))
    test_perform = pd.DataFrame({"Long": y_long, "DTC": y_dtc})
    plt.plot(y_test_dates, test_perform["Long"], "r", label="Long")
    plt.plot(y_test_dates, test_perform["DTC"], "g", label="DTC")
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Year')
    plt.ylabel('CumSum Return')
    plt.savefig('classifier.svg', dpi=1000)
    plt.show()

    return accuracies, test_perform


# In[ ]:




