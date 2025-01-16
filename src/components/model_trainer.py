import os  # Importing the os module to interact with the operating system, such as file path handling
import sys  # Importing the sys module for system-specific parameters and exceptions

from dataclasses import dataclass  # Importing the dataclass decorator for easy class creation with data attributes

# Importing various machine learning models for regression tasks
from catboost import CatBoostRegressor  # Importing CatBoost Regressor for gradient boosting-based regression
from sklearn.ensemble import (
    AdaBoostRegressor,  # AdaBoost Regressor for boosting algorithm-based regression
    GradientBoostingRegressor,  # Gradient Boosting Regressor for boosting algorithm-based regression
    RandomForestRegressor,  # Random Forest Regressor for ensemble-based regression
)
from sklearn.linear_model import LinearRegression  # Linear Regression model for basic regression
from sklearn.metrics import r2_score  # Importing r2_score to evaluate model performance
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors Regressor (not used here)
from sklearn.tree import DecisionTreeRegressor  # Decision Tree Regressor model for regression
from xgboost import XGBRegressor  # XGBoost Regressor for optimized gradient boosting

# Importing custom exception and logger
from src.exception import CustomException  # Custom exception for better error handling
from src.logger import logging  # Custom logging utility for logging important information during execution

# Importing utility functions for saving model objects and evaluating models
from src.utils import save_object, evaluate_models

# Defining a dataclass to store the configuration for the model trainer
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Path to save the trained model object

# Class for handling the training process of models
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initialize the configuration class for saving model

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")  # Log the start of data splitting
            # Split the training and testing datasets into features (X) and target labels (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Select all columns except the last one as features for training
                train_array[:, -1],  # Select the last column as the target variable for training
                test_array[:, :-1],  # Select all columns except the last one as features for testing
                test_array[:, -1],  # Select the last column as the target variable for testing
            )

            # Define a dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # Disable CatBoost output
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grid for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Hyperparameters for Decision Tree
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],  # Hyperparameters for Random Forest (number of trees)
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],  # Hyperparameters for Gradient Boosting (learning rate)
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # Fraction of samples used in each iteration
                    'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting iterations
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],  # Depth of the trees for CatBoost
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],  # Hyperparameters for AdaBoost
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                }
            }

            # Evaluate models using the evaluate_models function
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            # Find the best model based on the highest score from the model report
            best_model_score = max(sorted(model_report.values()))  # Highest model score

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            
            best_model = models[best_model_name]

            # If the best model score is below a threshold (e.g., 0.6), raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model
            predicted = best_model.predict(X_test)

            # Calculate the r2_score of the predictions
            r2_square = r2_score(y_test, predicted)

            # Return the R2 score
            return r2_square

        except Exception as e:
            # If any exception occurs, raise a custom exception with the error details
            raise CustomException(e, sys)
