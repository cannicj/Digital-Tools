what_factors_predict_sp500
==============================

This projects investigates which factors are useful in predicting S&P 500 returns. But what factors and what methods do we use to check the predictive power? After a lot of discussions within our group, we decided to focus on currency data as factors, namely the USD exchange rate to the currencies with the most volume. The goal is to find out whether currency data represents any information that is not yet included in the S&P 500.
There is no doubt that currency exchange rates reflect information about the economic state of a country, with inflation and interest rate expectations, import and export activity, etc.
Under the efficiency hypothesis, the same is the case for stock prices and effectively also the S&P500 index. Every new bit of information is immediately reflected in the data, even insider information. However, we use this chance to investigate whether there is some difference in how fast new information is reflected in the price data. To be able to also analyse potential nonlinear relations, we decided to focus on three different supervised machine learning techniques, namely the decisiontree classifier, the randomforest classifier as well as support vector machines. The models take the currency returns as inputs and extract a signal to buy or sell the S&P500. We keep the architecture of the models as flexible as possible, so we can play around with them to study the behavior and gain insights.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Required environment variables:
RESEARCH_PATH = "the path to your project folder"

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
