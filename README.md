# Disaster Response Project
## Summary

In event of a disaster, messages sent by victims can contain vital information needed to save lives. Categorizing these messages would make responses more effective since they can be directed to the responsible agencies involved in providing relief. This project used real data sets containing real messages sent during a disaster, to create an app which would enable emergency worker can input a new message and get classification of the message into categories related to or not related to the rescue or relief operation.

## Project Components
This project has three components.

1. ETL Pipeline

The ETL pipeline is implemented in a Python script, process_data.py. Here there messages and the category data sets are loaded and merged. Finally, the data is cleaned, deduplicated and stored in a SQLite database - DisasterResponse.db

2. ML Pipeline

The machine learning pipeline is implemented here in train_classifier.py. The data is first loaded from the the SQLite database and the split into training and test sets. The machine learning model is then built with a text processing and machine learning pipeline. The model is further tuned using GridSearchCV. Finally, the model is evaluated and exported as a picle file - classifier.pkl.

3. Flask Web App

A flask web app that provide a user interface for messages input and shows the category using the model.

## Project Execution Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database use the command below:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves use the command below:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Access the web app on your browser using http://0.0.0.0:3000/

### Credits
The following sources were consulted for this project:

- Udacity Data Science Nanodegree Class Room Solutions
