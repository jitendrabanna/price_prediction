
import os
import sys
import pandas as pd
import numpy as np
from typing import Any, cast

from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.sparse import issparse
from src.Airbnb.exception import customexception
from src.Airbnb.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.Airbnb.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','Preprocessor.pkl')


class DataTransformation:
    numerical_cols = ['amenities','accommodates','bathrooms','latitude','longitude','host_response_rate','number_of_reviews','review_scores_rating','bedrooms','beds']
    categorical_cols = ['property_type','room_type','bed_type','cancellation_policy','cleaning_fee','city','host_identity_verified','instant_bookable','host_has_profile_pic']

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    @staticmethod
    def _clean_host_response_rate(series: pd.Series) -> pd.Series:
        cleaned_series = series.astype('string').str.replace('%', '', regex=False)
        return pd.to_numeric(cleaned_series, errors='coerce')

    @staticmethod
    def _count_amenities(value) -> int:
        if pd.isna(value):
            return 0

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return 0
            return int(value)

        cleaned_value = str(value).strip()
        if cleaned_value in {'', '{}'}:
            return 0

        stripped_value = cleaned_value.strip('{}')
        if not stripped_value:
            return 0

        return len([item for item in stripped_value.split(',') if item.strip()])

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        prepared_df = dataframe.copy()
        prepared_df['host_response_rate'] = self._clean_host_response_rate(prepared_df['host_response_rate'])
        prepared_df['amenities'] = prepared_df['amenities'].apply(self._count_amenities)
        for column in self.categorical_cols:
            prepared_df[column] = prepared_df[column].astype(str).replace({'nan': np.nan, '<NA>': np.nan, 'None': np.nan})
        return prepared_df

    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')


            numerical_cols = self.numerical_cols
            categorical_cols = self.categorical_cols

            # Define which columns should be ordinal-encoded and which should be scaled
            property_type_cat = ['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft', 'Other', 'Guesthouse', 'Bed & Breakfast', 'Bungalow', 'Villa', 'Dorm', 'Guest suite', 'Camper/RV', 'Timeshare', 'Cabin', 'In-law', 'Hostel', 'Boutique hotel', 'Boat', 'Serviced apartment', 'Tent', 'Castle', 'Vacation home', 'Yurt', 'Hut', 'Treehouse', 'Chalet', 'Earth House', 'Tipi', 'Train', 'Cave', 'Casa particular', 'Parking Space', 'Lighthouse', 'Island']
            room_type_cat = ['Entire home/apt', 'Private room', 'Shared room']
            bed_type_cat = ['Real Bed', 'Futon', 'Pull-out Sofa', 'Airbed', 'Couch']
            # cancellation_policy_cat = ['strict', 'moderate', 'flexible', 'super_strict_30', 'super_strict_60'],
            cancellation_policy_cat = ['strict', 'moderate', 'flexible', 'super_strict_30', 'super_strict_60']
            cleaning_fee_cat = ['False', 'True']
            city_cat = ['NYC', 'SF', 'DC', 'LA', 'Chicago', 'Boston']
            host_identity_verified_cat = ['t', 'f']
            instant_bookable_cat = ['t', 'f']
            host_has_profile_pic_cat = ['t', 'f']

            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            # Categorigal Pipeline
            # cat_pipeline=Pipeline(
            #     steps=[
            #     ('imputer',SimpleImputer(strategy='most_frequent')),
            #     ('ordinalencoder',OrdinalEncoder(categories=[property_type_cat, room_type_cat, bed_type_cat, cancellation_policy_cat, cleaning_fee_cat, city_cat, host_has_profile_pic_cat, host_identity_verified_cat, instant_bookable_cat])),
            #     ('scaler',StandardScaler())])
            cat_pipeline = Pipeline(
                                steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('ordinalencoder', OrdinalEncoder(
                                        categories=[
                                            property_type_cat,
                                            room_type_cat,
                                            bed_type_cat,
                                            cancellation_policy_cat,
                                            cleaning_fee_cat,
                                            city_cat,
                                            host_identity_verified_cat,
                                            instant_bookable_cat,
                                            host_has_profile_pic_cat
                                        ],
                                        handle_unknown='use_encoded_value',
                                        unknown_value=-1
                                    ))
                                ]
                            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()

            train_df = self._prepare_features(train_df)
            test_df = self._prepare_features(test_df)

            logging.info("Host Response Rate converted to int")
            
            target_column_name = 'log_price'
            drop_columns = [target_column_name,'id',"name","description","first_review","host_since","last_review","neighbourhood","thumbnail_url", "zipcode"]


            # input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            # target_feature_train_df=train_df[target_column_name]
            
            
            # input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            # target_feature_test_df=test_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Input Feature Train Dataframe Head : \n{input_feature_train_df.head().to_string()}')
            logging.info(f'Target Feature Train Dataframe Head : \n{target_feature_train_df.head().to_string()}')

            logging.info(f'Input Feature Test Dataframe Head : \n{input_feature_test_df.head().to_string()}')
            logging.info(f'Target Feature Test Dataframe Head : \n{target_feature_test_df.head().to_string()}')

            logging.info(f'{input_feature_train_df.dtypes}')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            logging.info("Applying preprocessing object on training and testing datasets.")

            if issparse(input_feature_train_arr):
                input_feature_train_arr = cast(Any, input_feature_train_arr).toarray()
            if issparse(input_feature_test_arr):
                input_feature_test_arr = cast(Any, input_feature_test_arr).toarray()

            input_feature_train_arr = cast(NDArray[Any], np.asarray(input_feature_train_arr))
            input_feature_test_arr = cast(NDArray[Any], np.asarray(input_feature_test_arr))

            train_arr = np.concatenate([input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)], axis=1)


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            
    