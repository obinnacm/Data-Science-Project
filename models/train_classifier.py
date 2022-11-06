import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Load dataset from SQLite database into a dataframe.
    
    Parameters:
        - database_filepath: File path to the location of SQLite database file.
        
    Returns:
        - X: Numpy array of feature values which are the messages.
        - Y: Numpy array of labels for the feature values. 
        - Categories_names: The column names of the labels.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='disaster_response', con=engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """
    Replace Url, then tokenize and lemmatize messages.
    
    Parameters:
        - text: Text to be tokenized.
        
    Returns: Cleaned and tokenized text.
    
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build machine learning pipeline and improve results using parameters and GridSearchCV.
    
    Parameters: None
    
    Returns: Improved model cv.
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=0), n_jobs=-1)) ])

    parameters = {'clf__estimator__n_estimators': [50, 100, 200]
                  ,'clf__estimator__learning_rate': [0.1, 1.0, 1.1]}
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model using classification_report.
    
    Parameters:
        - model : The model to evaluate.
        - X_test: Test feature from split.
        - Y_test: Test label classification from split.
        - category_names: Associated name for labels.
     
    Returns: None.
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

    
def save_model(model, model_filepath):
    """
    Save machine learning model as a pickle file.
    
    Parameters:
        - model: The model to be saved.
        - model_filepath: The file path to where the .pkl file is to be stored.
        
    Returns: None
    
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


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