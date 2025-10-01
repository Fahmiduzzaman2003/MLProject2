import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_data, test_data)
    logging.info("Preprocessor saved at %s", preprocessor_path)

    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best model R2 score: {r2_score:.4f}")
