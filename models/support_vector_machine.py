#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def support_vector_machine(dataframe, currencies=None, include_sp500=True,lag=1,train_size=0.75):

    # Selecting columns based on currencies if provided
    if currencies is not None:
        columns_to_include = [currency for currency in currencies if currency in dataframe.columns]
        dataframe = dataframe[['DATE'] + columns_to_include + ['SP500']]

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
    
    # Training the SVC model
    svc_clf = SVC(kernel="rbf", random_state=42)
    svc_clf.fit(X_train, y_train)
    y_pred_train_svc = svc_clf.predict(X_train)
    y_pred_test_svc = svc_clf.predict(X_test)
    
    # Changing all 0 to -1 for return calculation
    y_pred_test_svc[np.where(y_pred_test_svc == 0)] = -1

    # Calculating returns
    y1_ret = y1_exp
    y_bench = y1_ret[y_test.index[0] - lag:]
    y_long = np.cumsum(y_bench)
    y_svc = np.cumsum(y_bench * y_pred_test_svc)

    # Calculating accuracies
    accuracies = pd.DataFrame({
        "Classifiers": ["SVC"],
        "in sample": [accuracy_score(y_train, y_pred_train_svc)],
        "out of sample": [accuracy_score(y_test, y_pred_test_svc)]
    }).set_index('Classifiers')

    print(accuracies)

    # Plotting
    plt.figure(figsize=(14, 6))
    test_perform = pd.DataFrame({"Long": y_long, "SVC": y_svc})
    plt.plot(y_test_dates, test_perform["Long"], "r", label="Long")
    plt.plot(y_test_dates, test_perform["SVC"], "g", label="SVC")
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Year')
    plt.ylabel('CumSum Return')
    plt.savefig('classifier.svg', dpi=1000)
    plt.show()

    return accuracies, test_perform


# In[ ]:




