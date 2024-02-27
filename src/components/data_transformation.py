import sys
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        '''
        try:
            numerical_features = ['glucose','education','age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI','BPMeds', 'prevalentStroke','prevalentHyp', 'diabetes']
            categorical_features=['sex']

            # Create preprocessing pipeline
            numerical_pipeline=Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('encoder', OneHotEncoder())
            ])
            
            logging.info(f"numerical_features columns: {numerical_features}")
            logging.info(f"Numerical columns: {categorical_features}")

            preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


        
    def initiate_data_transformation(self,raw_data_path):
        try:
            data = pd.read_csv(raw_data_path)
            # numerical_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI','BPMeds', 'prevalentStroke','prevalentHyp', 'diabetes']
            # data.dropna(subset=numerical_features, inplace=True)
            # data = pd.read_csv(r'C:\CHD_30_Jan\notebook\Data\data.csv')
            Y = data['TenYearCHD']
            X = data.drop(columns=['TenYearCHD', 'diaBP', 'heartRate', 'id', 'is_smoking'], axis=1)
          
            logging.info("Read data completed")

            logging.info("Obtaining preprocessing object")
            # c=DataTransformation()
            # preprocessing_obj = c.get_data_transformer_object()
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Split preprocessed data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
            # # Optionally, you can apply preprocessing separately to train and test sets
            X_train_preprocessed = preprocessing_obj.fit_transform(X_train)
            X_test_preprocessed = preprocessing_obj.transform(X_test)

            input_feature_train_df = X_train_preprocessed
            target_feature_train_df = y_train

            input_feature_test_df = X_test_preprocessed
            target_feature_test_df = y_test

            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )



            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
# c=DataTransformation()
# c.initiate_data_transformation(r'C:\CHD_30_Jan\notebook\Data\data.csv')