from IMDB_Project.exception import CustomeException
from IMDB_Project.logger import log

from IMDB_Project.components.data_ingestion import DataIngestion

from IMDB_Project.components.data_processing import preprocess_data

from IMDB_Project.components.model_trainer import ModelTraining

from IMDB_Project.components.model_evaluation import ModelEvaluator

import sys


def data_ingestion():
    log.info("--- Data Ingestion Started ----")
    obj = DataIngestion()
    obj.read_data()
    log.info("--- Data Ingestion Completed ---")



def data_preprocessing():
    log.info("----- Data Preprocessing Started -----")
    preprocess = preprocess_data()
    preprocess.preprocessing_data()
    preprocess.splitting_data()
    log.info("----- Preprocessing Completed ------")



def model_trainer():
    log.info("----- Model Training Started -----")
    param_tuner = ModelTraining()
    param_tuner.splitting()
    param_tuner.text_to_vectors()
    param_tuner.Hyperparameter_Tuning()
    log.info("----- Model Training Completed -----")


def model_evaluation():
    log.info("----- Model Evaluation Started -----")
    evaluator = ModelEvaluator()
    evaluator.evaluate()
    log.info("----- Model Evaluation Completed -----")


def training_pipeline():
    try:
        data_ingestion()
        data_preprocessing()
        model_trainer()
        model_evaluation()
    except Exception as e:
        raise CustomeException(e,sys)

if __name__=='__main__':
    print(" --------- Trainging Pipeline Started Checks Logs ----------\n")
    log.info("----- Training Pipeline Started -----")
    training_pipeline()
    print("\n--------- Trainging Pipeline Completed Checks Logs ----------\n")
















