# what_factors_predict_sp500
==============================

- [Introduction](#Introduction)
- [Project Organization](#Project-Organization)
- [Project flow](#Project-flow)
- [Data](#Data)
- [Models](#Models)
- [Docker set-up](#Docker-set-up)
- [Resources](#Resources)
- [Contributors](#Contributors)


## Introduction
------------
This projects aims to investigate what factors are useful in predicting S&P 500 returns. After extensive discussions within our group, we have chosen to concentrate our focus on currency data, specifically the USD exchange rate against the most traded currencies. The goal is to determine whether currency data can provide insights that are not already captured in the S&P 500, and therefore whether it acts as a predictor of S&P 500 returns.
It is widely known that currency exchange rates reflect information about the economic state of a country, with inflation and interest rate expectations, import and export activity etc. all having a real time impact on exchange rates.
Under the efficienct market hypothesis, this information contained in exchange rate data, should also be contained and priced into the S&P 500 index. Every new piece of information, even insider information, should immediately be incorporated into S&P 500 prices. However, we use this chance to investigate whether there exists some innefficiencies, and whether the above statement holds valid in the real world. Thus, we test whether fluctuations in exchange rates are able to provide a real time signal to future changes in the S&P 500. To be able to also analyse potential nonlinear relations, we have decided to focus on three different supervised machine learning techniques, namely the decisiontree classifier, the randomforest classifier and support vector machines. The models take currency returns as inputs and extract a signal to buy or sell the S&P500. We keep the architecture of the models as flexible as possible, so one can play around with them to study their behavior to gain insights.

## Project Organization --> needs to be updated in the end!
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


## Project flow
--------
- Set up skeletton project structure using cookiecutter.

- Set up the Docker to ensure reproducability.

- Set up functions to download data, using suitable APIs (FRED, YahooFinance, ...). Save data and process it to bring it into a format that can be used in the analysis (make logreturns).

- Write the three models and set them up as reusable functions. Describe input variables, define output.

- Test the models in different settings and play around with them, create an analysis notebook with different analysis steps. The notebook is the basis for the report.

- Create a playground for the user, such that everyone can play around with our models using ipywidgets.

- Writeup our findings in a structured report using latex.

- Make our finding presentable in a beamer presentation.


## Data 
------------

We pull currency exchange rates from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/) using the [pandas-datareader](https://github.com/pydata/pandas-datareader) python package.

S&P500 index data is obtained from [Yahoo Finance](https://finance.yahoo.com) using the [yfinance](https://pypi.org/project/yfinance/) python package.


## Models
------------

The models we use are all from scikit-learn, where they are all described in detail. 

- Support Vecotor machine: https://scikit-learn.org/stable/modules/svm.html. Parameter descriptions and further possibilities can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html.

- Decision Tree classifier: https://scikit-learn.org/stable/modules/tree.html. Parameter descriptions and further possibilities can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.

- Random Forest classifier: https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles. Parameter descriptions and further possibilities can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.


## Docker set-up (BASH)
------------
1. Make sure docker is installed on your system. It can be downloaded [here](https://www.docker.com/products/docker-desktop/)
2. Clone the github repository to your local machine
3. Build the docker file using the following bash command: docker build -t [name of image] .
4. Run the docker container: docker run -p 8888:8888 [name of image]
5. Copy the token given in the terminal when the container is run
6. Access http://localhost:8888/ and enter the token in the "Password or token" field
7. Explore the repository or run the notebooks



## Resources
------------
Add some sources that we can use in the bibliography.


## Contributors
------------

Jannic Cavegn (@cannicj), Alexander Falk (@AlexanderFalkETH), Julius Raschke (@juliusraschke)

--------


Required environment variables:
RESEARCH_PATH = "the path to your project folder"

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
