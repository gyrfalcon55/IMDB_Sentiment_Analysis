from IMDB_Project.logger import log
from IMDB_Project.components.model_prediction import Model_prediction



def prediction_pipeline():
    pred = Model_prediction()
    pred.load_files()
    pred.preprocessing_input()
    pred.prediction()

if __name__=='__main__':
    print(" --------- Prediction Pipeline Started Checks Logs ----------\n")
    log.info("Model prediction Started")
    prediction_pipeline()
    log.info("Model prediction Completed")
    print("\n--------- Prediction Pipeline Completed Checks Logs ----------\n")