import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

#from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            cat_columns = ['Building Type', 'Day of Week']
            num_columns = ['Square Footage', 'Number of Occupants', 'Appliances Used','Average Temperature']
            logging.info(f'Categorical Columns : {cat_columns}')
            logging.info(f'Numerical Columns : {num_columns}')
            # Define the custom ranking for each ordinal variable
            build_columns = ['Residential','Commercial','Industrial']
            day_columns = ['Weekday','Weekend']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_col= Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
            ])
            cat_col= Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=[build_columns,day_columns])),
                ('scaler',StandardScaler())
            ])
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_col',num_col,num_columns),
                    ('cat_col',cat_col,cat_columns)
                ]
            )
                    
            logging.info('Pipeline Completed')
            return preprocessor
           

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Energy Consumption'
            input_feature_train_df = train_df
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column_name]
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
           

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        

# if __name__=='__main__':
#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)