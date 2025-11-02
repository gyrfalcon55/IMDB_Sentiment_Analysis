from IMDB_Project.logger import log
from IMDB_Project.components.model_prediction import Model_prediction



def prediction_pipeline(user_input):
    pred = Model_prediction()
    pred.load_files()
    result = pred.prediction(user_input)   
    return result


if __name__=='__main__':
    print(" --------- Prediction Pipeline Started Checks Logs ----------\n")
    log.info("Model prediction Started")
    print("\n--------- Prediction Pipeline Completed Checks Logs ----------\n")