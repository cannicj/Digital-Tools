what_factors_predict_sp500
==============================

This projects aims to investigate what factors are useful in predicting S&P 500 returns. After extensive discussions within our group, we have chosen to concentrate our focus on currency data, specifically the USD exchange rate against the most heavily traded currencies. The goal is to determine whether currency data can provide insights not already captured in the S&P 500, and therefore whether it acts as a predictor of S&P 500 returns.
It is widely known that currency exchange rates reflect information about the economic state of a country, with inflation and interest rate expectations, import and export activity etc. all having a real time impact on exchange rates.
Under the efficienct market hypothesis, this information contained in exchange rate data, should also be contained and priced into the S&P 500 index. Every new piece of information, even insider information, should immediately be incorporated into S&P 500 prices. However, we use this chance to investigate whether there exists some innefficiencies, and whether the above statement holds valid in the real world. Thus, we test whether flutuations in exchange rates are able to provide a real time signal to future changes in the S&P 500. To be able to also analyse potential nonlinear relations, we have decided to focus on three different supervised machine learning techniques, namely the decisiontree classifier, the randomforest classifier and support vector machines. The models take currency returns as inputs and extract a signal to buy or sell the S&P500. We keep the architecture of the models as flexible as possible, so one can play around with them to study their behavior to gain insights.

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
    │                         `1.0-jja-initial-data-exploration`.
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
