import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            # 'VB' --> please, help, give 'VBP' --> do, are, have
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
            
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
class VerbHelpExtractor(BaseEstimator, TransformerMixin):
    def verb_help(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            tokenized_text = nltk.word_tokenize(sentence)
            
            for text in tokenized_text:
                if text == 'help' or  text == 'HELP':
                    return 1    
        
        return 0
    
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verb_help)
        return pd.DataFrame(X_tagged)
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('CleanDataset', engine)               
    df = df.replace(2,1)    
    
    # messages
    X = df['message']
    # 36 categories in the dataset
    category_names = df.columns[5:]
    y = df[category_names]
        
    return X, y, category_names

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('starting_verb', StartingVerbExtractor()),

            ('verb_help', VerbHelpExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
#     print(pipeline.get_params())
     
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (1, 0),
        'clf': [MultiOutputClassifier(RandomForestClassifier())],
        'clf__estimator__random_state':[42],
        'clf__estimator__n_estimators':[10, 36, 40],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def display_results(X_test, y_test, y_pred, model, category_names):
    raport = classification_report(y_test, y_pred, target_names=category_names)
    print("Classification report:\n", raport)
    score = model.score(X_test, y_test)
    print("Accuracy:", score)
    labels = np.unique(y_pred)
    print("Labels:", labels)

def save_params_to_file(best_params):
    print('Saving best parameters...\n    PARAMETERS: data/best_params.pkl')
    with open('data/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    print('Parameters saved!')
    
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    display_results(X_test, Y_test, y_pred, model, category_names)
    print("\nBest Parameters:", model.best_params_)
    # save best parameters into file
    save_params_to_file(model.best_params_)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Training model DONE')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()