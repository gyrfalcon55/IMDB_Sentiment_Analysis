from IMDB_Project.logger import log
from IMDB_Project.components.data_ingestion import DataIngestion
from IMDB_Project.utils.common import print_data
from IMDB_Project.exception import CustomeException
import pandas as pd
from pandas import DataFrame
import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
import os

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class preprocess_data:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.df = self.ingestion.read_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))


    def clean_text(self,text) -> str:
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and apply stemming
        filtered_words = [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stop_words]
        
        # Join back into a string
        return ' '.join(filtered_words)
    

    def preprocessing_data(self) -> DataFrame:
        log.info("Preprocessing data - converting to lowercase, removing special characters, word_tokenization, applying stemming (PorterStemmer)")
        try:
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'[^a-z A-Z 0-9\s]', '', x))

            self.df['text'] = self.df['text'].apply(self.clean_text)
        except Exception as e:
            raise CustomeException(e,sys)

    
    def splitting_data(self):
        try:
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
            log.info("Data splitted and saved to train_data.csv and test_data.csv")

            log.info("Creating train_data (data/processed/train_data) and test_data (data/processed/test_data) folders")
            os.makedirs(self.ingestion.var['data_dir']['train_data_dir'], exist_ok=True)
            os.makedirs(self.ingestion.var['data_dir']['test_data_dir'], exist_ok=True)

            log.info("Splitting dataset into train_data and test_data and saving inside data folder")
            train_df.to_csv(self.ingestion.var['data_dir']['train_data'], index=False)
            test_df.to_csv(self.ingestion.var['data_dir']['test_data'], index=False)

        except Exception as e:
            raise CustomeException(e, sys)


# if __name__=="__main__":
#     preprocess = preprocess_data()
#     preprocess.preprocessing_data()
#     preprocess.splitting_data()