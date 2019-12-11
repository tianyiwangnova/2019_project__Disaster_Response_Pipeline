# import packages
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, 
                            precision_score, recall_score, make_scorer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

from .etl_pipeline import etl
from .ml_pipeline import tokenize, average_f1_score


def load_data(messages_file_path="messages.csv",
              categories_file_path="categories.csv",
              table_name="message_table"):

    """
    Read, clean and load data;
    Output:
        X: an array of the messages
        y: a pandas dataframe of the category matrix
    """

    # read in file
    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    df = messages.merge(categories, on="id")

    # clean data
    df = etl(df)

    # load to database
    engine = create_engine('sqlite:///message_classification.db')
    df.to_sql(table_name, engine, index=False, if_exists="replace")

    # load data from database
    engine = create_engine('sqlite:///message_classification.db')
    df = pd.read_sql_table(table_name, con=engine)

    # define features and label arrays
    X = df['message']
    y = df[list(df.columns[4:])]

    return X, y


def build_model(clf=RandomForestClassifier(),
                gridsearch_params={},
                n_folds=3
                ):


    # text processing and model pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # create gridsearch object and return as final model pipeline
    f1_scorer = make_scorer(average_f1_score, greater_is_better=True)
    cv = GridSearchCV(pipeline, 
                      gridsearch_params, 
                      scoring=f1_scorer, 
                      n_jobs=-1, 
                      cv=n_folds, 
                      verbose=2)

    return cv


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # fit model
    begin = datetime.datetime.now()
    model.fit(X_train, y_train)
    time_pass = (datetime.datetime.now() - begin).seconds / 60
    print("running time: {:.2} min".format(time_pass))
    print("Best params: {}".format(model.best_params_))
    print("Highest averge f1 score: {}".format(cv_new.best_score_))

    # output model test results
    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[1]):
        print(classification_report(np.array(y_test)[i,:], y_pred[i,:]))

    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
