# tokenize the messages
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

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
    Calculates f1 scores for all labels and calculate the average
    """

    f1_scores = []
    for i in range(y_pred.shape[1]):
        f1_scores.append(f1_score(np.array(y_test)[:,i], y_pred[:,i],pos_label='1'))
    return np.mean(f1_scores)