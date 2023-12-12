# what_factors_predict_sp500
==============================

- [Introduction](#Introduction)
- [Project Organization](#Project-Organization)
- [Project flow](#Project-flow)
- [Data](#Data)
- [Models](#Models)
- [Docker set-up in BASH](#Docker-set-up-in-BASH)
- [Resources](#Resources)
- [Contributors](#Contributors)
- [Appendix](#Appendix)


## Introduction
------------
This projects aims to investigate what factors are useful in predicting S&P 500 returns. After extensive discussions within our group, we have chosen to concentrate our focus on currency data, specifically the USD exchange rate against the most traded currencies. The goal is to determine whether currency data can provide insights that are not already captured in the S&P 500, and therefore whether it acts as a predictor of S&P 500 returns.
It is widely known that currency exchange rates reflect information about the economic state of a country, with inflation and interest rate expectations, import and export activity etc. all having a real time impact on exchange rates.
Under the efficienct market hypothesis, this information contained in exchange rate data, should also be contained and priced into the S&P 500 index. Every new piece of information, even insider information, should immediately be incorporated into S&P 500 prices. However, we use this chance to investigate whether there exists some innefficiencies, and whether the above statement holds valid in the real world. Thus, we test whether fluctuations in exchange rates are able to provide a real time signal to future changes in the S&P 500. To be able to also analyse potential nonlinear relations, we have decided to focus on three different supervised machine learning techniques, namely the decisiontree classifier, the randomforest classifier and support vector machines. The models take currency returns as inputs and extract a signal to buy or sell the S&P500. We keep the architecture of the models as flexible as possible, so one can play around with them to study their behavior to gain insights.

## Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project	
    ├── .idea
    │   ├── inspectionProfiles						
    │   ├── .gitignore									
    │   ├── misc.xml									
    │   ├── modules.xml								
    │   ├── vcs.xml										
    │   └── what_factors_predict_sp500.iml			
    ├── data
    │   ├── cache/joblib								<- All files from the caching are saved here.
    │   ├── processed									<- Storing of processed files
    │   └── raw											<- Storing of raw downloaded files
    ├── notebooks
    │   ├── playground									<- playground to play with the models
    │   └── analysis									<- notebook with the full analysis
    ├── reports
    │   ├── figures										<- figures used in the report and beamer presentation
    │   ├── 0.0_main.tex								<- main Latex file
    │   ├── 0.1_titlepage.tex							<- storing of the titlepage
    │   ├── 1_File.tex									<- storing of the main content of the report
    │   ├── 2_references.bib							<- references for the bibliography
    │   ├── Presentation.tex							<- beamer presentation
    │   └── packages.sty								<- big file with all packages
    ├── src
    │   ├── data										<- all functions (.py) to downoad the data
    │   ├── features									<- all functions (.py) to build the features
    │   ├── models										<- all functions (.py) to run the models
    │   └── visualizations							<- all functions (.py) to visualize the results and for the playground
    ├── .gitignore										<- gitignore file with files to ignore
    ├── Dockerfile																				
    ├── Makefile											
    ├── requirements.txt								<- The requirements file for reproducing the analysis environment
    ├── setup.py											<- makes project pip installable (pip install -e .) so src can be imported												

## Project flow
--------
- Set up skeleton project structure using cookiecutter.

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


## Docker set-up in BASH
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
- [Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992)](https://doi.org/10.1145/130385.130401): A training algorithm for optimal
margin classifiers, Proceedings of the Fifth Annual Workshop on Computa-
tional Learning Theory, 144–152.
- [Breiman, L. (1984)](https://doi.org/10.1201/9781315139470): Classification and regression trees, Routledge.
- [Ho, T. K. (1995)](https://doi.org/10.1109/ICDAR.1995.598994): Random decision forests. Proceedings of 3rd International Con-
ference on Document Analysis and Recognition, 1, 278–282 vol.1




## Contributors
------------

Jannic Cavegn (@cannicj), Alexander Falk (@AlexanderFalkETH), Julius Raschke (@juliusraschke)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Appendix
------------

Note that the report is only compiling correcly when using a ‘real’ pdf reader like acrobat, Microsoft edge cannot handle animations in pdf files (it just shows a blurry image).
