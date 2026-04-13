import sys
from pathlib import Path
from typing import Union

import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import load_object
from src.Airbnb.exception import customexception
from src.Airbnb.components.Data_transformation import DataTransformation
from src.Airbnb.components.Model_trainer import ModelTrainer


class PredictPipeline:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifacts_dir = self.project_root / 'Artifacts'
        self.preprocessor_path = self.artifacts_dir / 'Preprocessor.pkl'
        self.model_path = self.artifacts_dir / 'Model.pkl'
        self.train_data_path = self.artifacts_dir / 'train_data.csv'
        self.test_data_path = self.artifacts_dir / 'test_data.csv'

    def _ensure_prediction_artifacts(self):
        if self.preprocessor_path.exists() and self.model_path.exists():
            return

        if not self.train_data_path.exists() or not self.test_data_path.exists():
            raise FileNotFoundError(
                'Prediction artifacts are missing. Expected train_data.csv and test_data.csv under Artifacts/ to rebuild the model.'
            )

        logging.info('Model artifact missing. Rebuilding prediction artifacts from saved train/test datasets.')
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initialize_data_transformation(
            str(self.train_data_path),
            str(self.test_data_path),
        )
        model_trainer = ModelTrainer()
        model_trainer.initate_model_training(train_arr, test_arr)
        logging.info('Prediction artifacts rebuilt successfully.')
    
    def predict(self, features):
        try:
            self._ensure_prediction_artifacts()
            preprocessor = load_object(str(self.preprocessor_path))
            model = load_object(str(self.model_path))
            logging.info('Preprocessor and Model Pickle files loaded')

            prepared_features = DataTransformation()._prepare_features(features)
            scaled_data = preprocessor.transform(prepared_features)
            logging.info('Data Scaled')
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(str(e), sys)

class CustomData:
    def __init__(self,
                 property_type: str,
                 room_type: str,
                 amenities: Union[str, int],
                 accommodates: Union[str, int],
                 bathrooms: Union[str, int, float],
                 bed_type: str,
                 cancellation_policy: str,
                 cleaning_fee: str,
                 city: str,
                 host_has_profile_pic: str,
                 host_identity_verified: str,
                 host_response_rate: str,
                 instant_bookable: str,
                 latitude: Union[str, float],
                 longitude: Union[str, float],
                 number_of_reviews: Union[str, int],
                 review_scores_rating: Union[str, int, float],
                 bedrooms: Union[str, int, float],
                 beds: Union[str, int, float]):
        
        self.property_type = property_type
        self.room_type = room_type
        self.amenities = amenities
        self.accommodates = accommodates
        self.bathrooms = bathrooms
        self.bed_type = bed_type
        self.cancellation_policy = cancellation_policy
        self.cleaning_fee = cleaning_fee
        self.city = city
        self.host_has_profile_pic = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.host_response_rate = host_response_rate
        self.instant_bookable = instant_bookable
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_reviews = number_of_reviews
        self.review_scores_rating = review_scores_rating
        self.bedrooms = bedrooms
        self.beds = beds

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'property_type': [self.property_type],
                'room_type': [self.room_type],
                'amenities': [int(self.amenities)],
                'accommodates': [int(self.accommodates)],
                'bathrooms': [float(self.bathrooms)],
                'bed_type': [self.bed_type],
                'cancellation_policy': [self.cancellation_policy],
                'cleaning_fee': [self.cleaning_fee],
                'city': [self.city],
                'host_has_profile_pic': [self.host_has_profile_pic],
                'host_identity_verified': [self.host_identity_verified],
                'host_response_rate': [self.host_response_rate],
                'instant_bookable': [self.instant_bookable],
                'latitude': [float(self.latitude)],
                'longitude': [float(self.longitude)],
                'number_of_reviews': [int(self.number_of_reviews)],
                'review_scores_rating': [float(self.review_scores_rating)],
                'bedrooms': [float(self.bedrooms)],
                'beds': [float(self.beds)]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(str(e), sys)
