# Disaster Response Pipeline

*Tianyi Wang*
*2019 Dec 15th*

This project is a pipeline to clean and tokenize messages sent during disaster events, fit machine learning model (random forest classifier) on the messages and their categories (so the model can classify new messages), and present the analysis on the data and a small message classification application on a website. This project is one of Udacity Data Scientist Nanodegree projects.

## Overview

The machine learning pipeline is based on data like this:

![data](https://raw.githubusercontent.com/tianyiwangnova/2019_project__Disaster_Response_Pipeline/master/screenshots/data.png)

Every message sent may belong to one or more categories. There are many categories:

![cat](https://raw.githubusercontent.com/tianyiwangnova/2019_project__Disaster_Response_Pipeline/master/screenshots/categories.png)

Messages and categories are stored in different files. We first combined these two tables and did some data cleaning and stored the table to a [sqlite](https://docs.python.org/3/library/sqlite3.html) databsed. Then we used `MultiOutputClassifie`r and `RandomForestClassifier` in [sklearn](https://scikit-learn.org/stable/) to fit the messages and categories data. 

When tuning the model, we used grid search method on a variety of parameters including `min_samples_split`, `min_samples_leaf` and `max_features`. To handle the data imblanace (some categories have much fewer training samples), we used **f1_score** as the scoing function and we actually saw that model with higher f1 score also has higher accuracy. After we finished grid searching and got the fitted model, we stored the model as a pickle file.

The output of the analysis will be fed to the *Disaster Response Project* web app template. On the homepage we will display some visualizations to present an overview of the training dataset:

![homepage](https://raw.githubusercontent.com/tianyiwangnova/2019_project__Disaster_Response_Pipeline/master/screenshots/homepage.png)

You can enter a message to classify:

![classify](https://raw.githubusercontent.com/tianyiwangnova/2019_project__Disaster_Response_Pipeline/master/screenshots/query.png)

## Files/folders in this repo:

  - **data**: folder where the messages and categories data are stored
  - **templates**: html templates of the web app
  - **message_classification.db**: sqllite_database where the combined table of the messages and categories sits
  - **model**: we have trained the model (test average f1 scores among categories is 0.2612; test average accuracy: 95.86%) so you can directly run the web app; but you can definitely retrain the model yourself
  - **run.py**: scipts for running the web app 
  - **train_model.py**: scripts for ETL, fitting machine learning models and saving the data and model

## How to use it?

To retrain the model, you can run `train_model.py` in the terminal

```sh
$ python train_model.py
```
or specify `n_jobs` yourself if you encounter this error `_pickle.PicklingError: Could not pickle the task to send it to the workers.`

```sh
$ python train_model.py --n_jobs=1
```

You can actually specify a lot of variables 
 - **messages_file_path**: file path of the message data
 - **categories_file_path**: file path of the categories data
 - **table_name**: name of the combined table; default: message_table
 - **gridsearch_params**: grid search parameters; defalt: `{'clf__estimator__min_samples_split': [5, 10, 15],
                                 'clf__estimator__min_samples_leaf': [1, 3, 5]}`
 - **n_folds**: numbers of cross validation folds; default: 5
 - **n_jobs**: number of jobs to run in parallel; default: 5
 
To run the app, you can run `run.py` in the terminal
```sh
$ python run.py
```
The web app will be hosted on `http://0.0.0.0:3002/`

If you specified a different `table_name` when you run `train_model.py`, you will also need to specify it here
```sh
$ python run.py --table_name <your_table_name>
```


