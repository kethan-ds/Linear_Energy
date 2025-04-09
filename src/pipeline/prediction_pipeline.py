import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,final_new_data):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(final_new_data)
            #data_scaled1=preprocessor(features)  #KETAN TEST
            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 building_type:str,
                 square_footage:int,
                 number_of_occupants:int,
                 appliances_used:int,
                 average_temperature:float,
                 day_of_week:str):
        
        self.building_type=building_type
        self.square_footage=square_footage
        self.number_of_occupants=number_of_occupants
        self.appliances_used=appliances_used
        self.average_temperature=average_temperature
        self.day_of_week = day_of_week

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Building Type':[self.building_type],
                'Square Footage':[self.square_footage],
                'Number of Occupants':[self.number_of_occupants],
                'Appliances Used':[self.appliances_used],
                'Average Temperature':[self.average_temperature],
                'Day of Week':[self.day_of_week]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
