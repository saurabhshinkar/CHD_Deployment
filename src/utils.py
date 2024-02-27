import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

from sqlalchemy import create_engine
import pandas as pd
import urllib.parse
username = 'postgres'
password = 'Saurabh@7'
host = 'localhost'
port = '5432'
database = 'postgres'
# Encode the password
encoded_password = urllib.parse.quote(password, safe='')

# Construct the connection string
connection_string = f'postgresql://{username}:{encoded_password}@{host}:{port}/{database}'

def get_data_from_sql(id):
    # Replace 'your_connection_string' with your actual connection string to PostgreSQL
    engine = connection_string

    # Replace 'your_table_name' with the name of your table in PostgreSQL
    sql_query = "SELECT * FROM chdtest4 WHERE id = {}".format(id)

    # Execute the SQL query and create a DataFrame
    pred_df = pd.read_sql(sql_query, engine)
    # pred_df=pred_df[['glucose','education','age', 'cigsperday', 'totchol', 'sysbp', 'bmi','sex']]
    pred_df=pred_df[['glucose','education','age', 'cigsperday', 'totchol', 'sysbp', 'bmi','sex','bpmeds','prevalenthyp','prevalentstroke','diabetes']]
    column_rename_mapping = {
    'cigsperday': 'cigsPerDay',
    'totchol': 'totChol',
    'sysbp': 'sysBP',
    'bmi':'BMI',
    'bpmeds':'BPMeds',
    'prevalenthyp':'prevalentHyp',
    'prevalentstroke':'prevalentStroke'}

    pred_df.rename(columns=column_rename_mapping, inplace=True)

    return pred_df