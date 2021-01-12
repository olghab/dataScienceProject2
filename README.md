# dataScienceProject2
## Disaster Response Pipeline Project

### Introduction:
The aim of this project is to analyze disaster responses data from Figure Eight company (https://appen.com/), by:
1) preparing dataset containing real messages recorded during disaster events
2) creating machine learning model for messeges classification into 36 categories, in order to send them to an appropriate disaster relief agency
3) visualising results in web app

The project is the part of Udacity Program - Data Science Nanodegree

### Instructions:
1. In order to set up the database and model, please run following lines in root directory:

    - To run ETL pipeline that cleans data and stores in database
        `process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. In order to run web app, please type following line in /app directory
    `python run.py`

3. To run the visualization on local machine, go to http://localhost:3001/

4. ----> to run it on Project Workspace IDE << instruction for myself >>
	- in new terminal type env|grep WORK
    - copy all ID starting with view....
    - in the browser type: https://view6914b2f4-3001.udacity-student-workspaces.com/, where view6914b2f4 is taken from GREP
    - press enter
 
### Project components:
 1. process_data.py - python script that cleans data, using ETL pipeline. It loads disaster_categories.csv and disaster_messages.csv files from /data directory, cleans them and saves in SQLite database
 
 2. models/train_classifier.py - python script that creates machine learning pipeline in order to classify data. It loads clean data from SQLite database, splits datasets into training and test sets, builds a text processing and machine learning pipeline and then trains model with tuned parameters found by GridSearchCV. Best parameters are saved in data/best_params.pkl file and final model is exported as pkl file, saved in /models directory.
 - Machine learning algorithm classifies data using MultiOutputClassifier with Random Forest Classifier as estimator 

3. Visualization of the results is prepared in /app directory that consists of run.py flask file that runs app, and master.html and go.html pages of the web app
