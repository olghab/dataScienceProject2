import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Decription:
    This function loads data from two csv files 
    and merge them into one dataframe with merged ID column
    Arguments:
        messages_filepath: path to disaster_messages.csv
        categories_filepath: path to disaster_categories.csv
    Returns:
        df - created dataframe
    """
    # load messages dataset 
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)  
    
    # merge datasets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')
    
    return df

def clean_data(df):
    """ 
    Decription:
    This function prepares data for further analysis
    Arguments:
        df - raw dataframe
    Returns:
        df - cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df["categories"]).str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    # extract a list of new column names for categories, using the first row
    category_colnames = []
    for name in row:
        category_colnames.append(name.split("-")[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #  Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        value = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = value.astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(keep=False, inplace=True)
    
    return df


def save_data(df, database_filename):
    """ # Save the clean dataset into an sqlite database """ 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CleanDataset', engine)  


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