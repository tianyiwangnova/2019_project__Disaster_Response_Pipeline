import argparse
import json
import plotly
import pandas as pd
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import *
from sklearn.externals import joblib

from train_model import *


app = Flask(__name__)


def tokenize(text):

    """
    clean and tokenize the text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#grab the table_name in the argument
parser=argparse.ArgumentParser()
parser.add_argument("--table_name", default="message_table", type=str)
args = parser.parse_args()

# load data
engine = create_engine('sqlite:///message_classification.db')
df = pd.read_sql_table(args.table_name, engine)

# load model
model = joblib.load("model")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    """
    Create visualizations with plotly and render the website with the plots
    """
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [

        #bar chart of the genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        #Bar chart of numbers of messages in each category
        {
            'data': [
                Bar(
                    x=list(df.iloc[:, 4:].astype("int").sum().sort_values(ascending=False).index),
                    y=list(df.iloc[:, 4:].astype("int").sum().sort_values(ascending=False).values)
                )
            ],

            'layout': {
                'title': 'Numbers of messages in each category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        #Heatmap chart of the correlation matrix of the categories
        {
            'data': [
                Heatmap(
                    x=list(df.drop('child_alone',axis=1).iloc[:, 7:].columns),
                    y=list(df.drop('child_alone',axis=1).iloc[:, 7:].columns),
                    z=np.array(df.drop('child_alone',axis=1).iloc[:, 7:].astype("int").corr().replace(1, np.nan)),
                    colorscale = 'Viridis'
                )
            ],

            'layout': {
                'title': 'Correlation matrix of the categories',
                'width':1000,
                'height':1000
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():

    """
    make prediction with the `query` the user inputs;
    render the '/go' page with the classfication result
    """

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():

    """
    the app will be run on http://0.0.0.0:3002/
    """

    app.run(host='0.0.0.0', port=3002, debug=True)


if __name__ == '__main__':
    main()