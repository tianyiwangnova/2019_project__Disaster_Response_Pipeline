import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


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
        categories[column] = categories[column].astype(int)

    #some column(s) have more than one unique values; we want to change that
    col_more_unique_values = list(categories.columns[categories.describe().loc['max'] > 1])
    for column in col_more_unique_values:
        categories[column] = categories[column].apply(lambda x: 1 if x > 0 else 0)

    #replace category column with the new category columns
    df = df.drop('categories', axis=1).join(categories)

    #drop duplicates
    df.drop_duplicates(inplace=True)

    return df


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("--messages_file_path", default="data/messages.csv", type=str)
    parser.add_argument("--categories_file_path", default="data/categories.csv", type=str)
    parser.add_argument("--database_file_path", default="sqlite:///message_classification.db", type=str)
    parser.add_argument("--table_name", default="message_table", type=str)

    args = parser.parse_args()

    messages_file_path = args.messages_file_path
    categories_file_path = args.categories_file_path
    table_name = args.table_name
    database_file_path = args.database_file_path

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    df = messages.merge(categories, on="id")

    df = etl(df)

    # load to database
    engine = create_engine(database_file_path)
    df.to_sql(table_name, engine, index=False, if_exists="replace")
