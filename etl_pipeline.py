import pandas as pd
from sqlalchemy import create_engine

#load data
messages = pd.read_csv("messages.csv")
categories = pd.read_csv("categories.csv")

#merge messages and categories
df = messages.merge(categories, on="id")

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
    categories[column] = categories[column].apply(lambda x: 1 if int(x) > 0 else 0)

#replace category column with the new category columns
df = df.drop('categories', axis=1).join(categories)

#drop duplicates
df.drop_duplicates(inplace=True)

#save the clean dataset to sqlite database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, index=False)

