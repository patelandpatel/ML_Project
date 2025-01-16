import sys  # Imports the sys module, which provides access to system-specific parameters and functions
from dataclasses import dataclass  # Imports the dataclass decorator to easily create classes used to store data

# Importing necessary libraries for data processing
import numpy as np  # Imports NumPy for working with arrays and numerical data
import pandas as pd  # Imports pandas for data manipulation and analysis (e.g., working with DataFrames)
from sklearn.compose import ColumnTransformer  # Imports ColumnTransformer to apply different preprocessing to specific columns
from sklearn.impute import SimpleImputer  # Imports SimpleImputer to fill missing values in the dataset
from sklearn.pipeline import Pipeline  # Imports Pipeline to bundle preprocessing steps into a single object
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Imports OneHotEncoder (for categorical encoding) and StandardScaler (for scaling numerical data)

# Custom imports for logging and exception handling
from src.exception import CustomException  # Custom exception class for handling errors
from src.logger import logging  # Custom logging module for logging important messages
import os  # Imports os module for interacting with the operating system, e.g., working with file paths

from src.utils import save_object  # Imports save_object function to save the preprocessor to a file

# A dataclass to hold configuration values related to data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")  # Specifies the file path where the preprocessing object will be saved

# Class for handling data transformation operations
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Initializes the configuration object for data transformation

    def get_data_transformer_object(self):
        '''
        This function is responsible for defining and returning the data transformation pipeline.
        It creates pipelines for both numerical and categorical columns.
        '''
        try:
            # Defining columns
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns in the dataset
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"
            ]  # List of categorical columns

            # Pipeline for numerical data processing
            num_pipeline = Pipeline(
                steps=[  # List of steps in the pipeline
                    ("imputer", SimpleImputer(strategy="median")),  # Imputes missing values using the median of the column
                    ("scaler", StandardScaler())  # Scales numerical features to have zero mean and unit variance
                ]
            )

            # Pipeline for categorical data processing
            cat_pipeline = Pipeline(
                steps=[  # List of steps in the pipeline
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fills missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Encodes categorical features as one-hot vectors
                    ("scaler", StandardScaler(with_mean=False))  # Scales categorical features without centering the data
                ]
            )

            # Logging column details for debugging purposes
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Creating a column transformer that applies the numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns
                    ("cat_pipelines", cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical columns
                ]
            )

            return preprocessor  # Returns the preprocessor object which combines both pipelines
        
        except Exception as e:
            raise CustomException(e, sys)  # If any error occurs, raise a custom exception with the error details

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function reads the training and testing datasets, applies the preprocessing steps,
        and returns the transformed training and testing arrays.
        '''
        try:
            # Reading the train and test CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Logging that the data has been read successfully
            logging.info("Read train and test data completed")

            # Obtaining the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Defining the target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Splitting the features and target columns for both train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Logging that preprocessing is being applied
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Applying the preprocessing pipeline to both training and testing feature sets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining the features and target into final train and test arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Logging that the preprocessing object has been saved
            logging.info(f"Saved preprocessing object.")

            # Saving the preprocessing object to a file for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Returning the transformed train and test arrays and the file path of the saved preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)  # If any error occurs, raise a custom exception with the error details




