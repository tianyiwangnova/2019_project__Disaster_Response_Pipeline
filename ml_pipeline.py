# import libraries
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('messages_new1', con=engine)
X = df['message']
Y = df[list(df.columns[4:])]

# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('messages_new1', con=engine)
X = df['message']
Y = df[list(df.columns[4:])]

# tokenize the messages
def tokenize(text):
    
    text = text.lower() #lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #remove punctuations
    
    tokens = word_tokenize(text) #tokenize
    lemmatizer = WordNetLemmatizer() 
    
    #remove stopwords
    clean_tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    #reduce words to their root form
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in clean_tokens]

    return clean_tokens

