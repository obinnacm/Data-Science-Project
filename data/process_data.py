import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories dataset from the csv files.
    
    Parameters:
        - messages_filepath: File path location of disaster_messages.csv.
        - categories_filepath: File path location of disaster_categories.csv.
        
    Returns: Dataframe of the merged data.
    
    """
    messages = pd.read_csv(messages_filepath,encoding='latin-1')
    categories = pd.read_csv(categories_filepath,encoding='latin-1')
    df = pd.merge(messages,categories, on="id",how="inner")
    return df


def clean_data(df):
    """
    Clean dataframe by spliting categories into columns and deduplicate.
    
    Parameters:
        - df: Merged disaster messages and categories dataframe.
        
    Returns: Cleaned and deduplicated dataset.
    
    """
    categories = df[["id","categories"]]
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [col[:len(col)-2] for col in row.tolist()]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype('int')
        categories.related.replace(2, 0, inplace=True)
    df = df.drop(columns="categories")
    df = pd.concat([df, categories], axis="columns")
    df = df.drop_duplicates('id')
    return df


def save_data(df, database_filename):
    """
    Save dataframe in SQLite database.
    
    Parameters:
        - df: Dataframe to save.
        - database_filename: Name of the database file to be saved.
        
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(name='disaster_response',con=engine, index=False, if_exists='replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()