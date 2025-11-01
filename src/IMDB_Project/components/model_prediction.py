from IMDB_Project.logger import log
from IMDB_Project.components.data_ingestion import DataIngestion
from IMDB_Project.utils.common import read_yml
from IMDB_Project.components.data_processing import preprocess_data
from IMDB_Project.exception import CustomeException
import sys
import joblib
import re

class input_preprocessing:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.preprocess = preprocess_data()
        try:
            self.saved_model_path = self.ingestion.var['saved_model_path']
            self.saved_vectorizer_path = self.ingestion.var['saved_vectorizer_path']
        except Exception as e:
            raise CustomeException(e,sys)

    def load_files(self):
        log.info("Loading Model and Vectorizer for predictions")
        self.model = joblib.load(self.saved_model_path)
        self.vectorizer = joblib.load(self.saved_vectorizer_path)

    def preprocessing_input(self):

        self.user_input = """Im a die hard Dads Army fan and nothing will ever change that. I got all the tapes, DVD's and audiobooks and every time i watch/listen to them its brand new. <br /><br />The film. The film is a re run of certain episodes, Man and the hour, Enemy within the gates, Battle School and numerous others with a different edge. Introduction of a new General instead of Captain Square was a brilliant move - especially when he wouldn't cash the cheque (something that is rarely done now).<br /><br />It follows through the early years of getting equipment and uniforms, starting up and training. All in all, its a great film for a boring Sunday afternoon. <br /><br />Two draw backs. One is the Germans bogus dodgy accents (come one, Germans cant pronounced the letter "W" like us) and Two The casting of Liz Frazer instead of the familiar Janet Davis. I like Liz in other films like the carry ons but she doesn't carry it correctly in this and Janet Davis would have been the better choice.
"""
                
                
        log.info("preprocessing hardcoded user_input and converting text to vector")
        self.new_output = re.sub(r'[^a-zA-Z0-9\s]', '', self.user_input)
        self.new_output = self.preprocess.clean_text(self.new_output)
        self.vector = self.vectorizer.transform([self.new_output])

class Model_prediction(input_preprocessing):

    def prediction(self):
        log.info("Model prediction on user_input")
        self.result = self.model.predict(self.vector)
        log.info("\n*************************************\n")
        if self.result[0] == 1:
            log.info("positive Sentiment")
            print("positive Sentiment")
        else:
            log.info("negative sentiment")
            print("negative sentiment")
        log.info("\n*************************************\n")   

# if __name__=='__main__':
#     obj = Model_prediction()
#     obj.load_files()
#     obj.preprocessing_input()
#     obj.prediction()



