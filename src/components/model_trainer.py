import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models: Dict[str, object] = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            params: Dict[str, Dict[str, object]] = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                },
                "Linear Regression": {},
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            if not model_report:
                raise CustomException("Model evaluation returned no results", sys)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                "Best model: %s with R2 %.4f",
                best_model_name,
                best_model_score,
            )

            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predictions = best_model.predict(X_test)
            final_score = r2_score(y_test, predictions)
            logging.info("Final R2 score on test data: %.4f", final_score)
            return final_score
        except Exception as e:
            raise CustomException(e, sys)
