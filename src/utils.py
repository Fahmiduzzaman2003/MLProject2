import os
import sys
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    """Persist a Python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Saved object to %s", file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Load a Python object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Train and evaluate multiple models, returning their test R2 scores."""
    try:
        report: Dict[str, float] = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            if param_grid:
                logging.info("Tuning hyperparameters for %s", model_name)
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                logging.info("Training %s without hyperparameter tuning", model_name)
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(
                "%s -> train R2: %.4f, test R2: %.4f",
                model_name,
                train_score,
                test_score,
            )
            report[model_name] = test_score
            models[model_name] = model
        return report
    except Exception as e:
        raise CustomException(e, sys)
