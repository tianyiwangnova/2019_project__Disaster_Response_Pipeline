# Disaster Response Pipeline

*Tianyi Wang*

This project is a pipeline to clean and tokenize messages sent during disaster events, fit machine learning model (random forest classifier) on the messages and their categories (so the model can classify new messages), and present the basic analysis on the data and a small message classification application on a website. This project is one of Udacity Data Scientist Nanodegree projects.

## Overview

The project is based on a dataset like this:

Every message sent may belong to one or more categories. There are many categories there:

Messages and categories are stored in different files. We first combined these two tables and did some data cleaning and stored the table to a [sqlite](https://docs.python.org/3/library/sqlite3.html) databsed. Then We used `MultiOutputClassifie`r and `RandomForestClassifier` in [sklearn](https://scikit-learn.org/stable/) to fit the messages and categories data. When tuning the model, we used grid search method on a variety of parameters including `min_samples_split`, `min_samples_leaf` and `max_features`. To handle the data imblanace (some categories have much fewer training samples), we used f1_score as the scoing function and when we actually saw that model with higher f1 score also has higher accuracy. After we finished grid searching and got the fitted model, we stored the model as a pickle file.

The output of the analysis will be fed to the *Disaster Response Project* website template. On the homepage we will display some visualizations to present an overview of the training dataset:
![homepage](https://raw.githubusercontent.com/tianyiwangnova/2019_project__Disaster_Response_Pipeline/master/screenshots/homepage.png)
