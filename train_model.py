import argparse
import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

import pickle

import warnings
warnings.filterwarnings('ignore')


def etl(df):

    """
    Data cleaning on the message dataset; 
    Input data should be the joint result of messages and categories
    Steps:
    (1) Clean category data to get a matrix of 0s and 1s
    (2) Drop duplicates
    """

    ##clean category field
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    #extract the columns names of the new category data
    row = categories.loc[0]
    category_colnames = categories.loc[0].apply(lambda x: x.split("-")[0]).values
    categories.columns = category_colnames

    #convert category numbers to just 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])  
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)

    #some column(s) have more than one unique values; we want to change that
    col_more_unique_values = list(categories.columns[categories.describe().loc['unique'] > 2])
    for column in col_more_unique_values:
        categories[column] = categories[column].apply(lambda x: "1" if int(x) > 0 else "0")

    #replace category column with the new category columns
    df = df.drop('categories', axis=1).join(categories)

    #drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def tokenize(text):

    """
    Tokenize function which will be used in CountVectorizer;
    Steps:
    (1) lower the case
    (2) remove punctuations
    (3) tokenize
    (4) remove stopwords
    (5) reduce words to their root form
    Output is a list of cleaned tokens
    """
    

    text = text.lower() #lower case
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    
    tokens = word_tokenize(text) #tokenize
    lemmatizer = WordNetLemmatizer() 
    
    #remove stopwords
    clean_tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    #reduce words to their root form
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]

    return clean_tokens


def average_f1_score(y_test, y_pred):

    """
    F1 scoring function for Grid Search;
    Calculates f1 scores for all labels and then calculate the average
    """

    f1_scores = []
    for i in range(y_pred.shape[1]):
        f1_scores.append(f1_score(np.array(y_test)[:,i], y_pred[:,i],pos_label='1'))
    return np.mean(f1_scores)

def load_data(messages_file_path, categories_file_path, table_name):

    """
    Read, clean and load data;
    The joint data (messages + categories) will be stored to 'sqlite:///message_classification.db';
    Need to specify the name of the saved table;
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


def build_model(gridsearch_params, n_folds, n_jobs):

    """
    Create a GridSearchCV object of the modeling pipeline
    Args:
    gridsearch_params: same as param_grid in GridSearchCV;
    n_folds: The number of cross-validation splits
    n_jobs: how many paralle jobs to run
    """

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
                      n_jobs=n_jobs,
                      cv=n_folds, 
                      verbose=2)

    return cv


def train(X, y, model):

    """
    Fit the GridSearch object.
    Args:
    X,y should be the output of function load_data;
    model should be the output of function build_model
    """

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # fit model
    begin = datetime.datetime.now()
    model.fit(X_train, y_train)
    time_pass = (datetime.datetime.now() - begin).seconds / 60
    print("running time: {:.2} min".format(time_pass))
    print("Best params: {}".format(model.best_params_))
    print("Highest averge f1 score: {}".format(model.best_score_))

    # output model test results
    y_pred = model.predict(X_test)
    
    #print the classification report for each category
    for i in range(y_pred.shape[1]):
        print(classification_report(np.array(y_test)[i,:], y_pred[i,:]))

    return model


def export_model(model):

    """
    save the fitted model as a pickle file with the name "model"
    """

    # Export model as a pickle file
    pickle.dump(model, open("model", 'wb'))


def run_pipeline(messages_file_path,
                 categories_file_path,
                 table_name,
                 gridsearch_params,
                 n_folds,
                 n_jobs):

    X, y = load_data(messages_file_path, categories_file_path, table_name)
    print("Shape of X: {}".format(X.shape))
    cv = build_model(gridsearch_params, n_folds, n_jobs)
    model = train(X, y, cv)
    export_model(model)


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("--messages_file_path", default="data/messages.csv", type=str)
    parser.add_argument("--categories_file_path", default="data/categories.csv", type=str)
    parser.add_argument("--table_name", default="message_table", type=str)
    parser.add_argument("--gridsearch_params", 
                        default={'clf__estimator__min_samples_split': [5, 10, 15],
                                 'clf__estimator__min_samples_leaf': [1, 3, 5]}, 
                        type=dict)
    parser.add_argument("--n_folds", default=5, type=int)
    parser.add_argument("--n_jobs", default=5, type=int)

    args = parser.parse_args()

    messages_file_path = args.messages_file_path
    categories_file_path = args.categories_file_path
    table_name = args.table_name
    gridsearch_params = args.gridsearch_params
    n_folds = args.n_folds
    n_jobs = args.n_jobs

    run_pipeline(messages_file_path,
                 categories_file_path,
                 table_name,
                 gridsearch_params,
                 n_folds,
                 n_jobs)