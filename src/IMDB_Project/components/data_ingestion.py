from IMDB_Project.logger import log
from IMDB_Project.utils.common import read_yml
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import os



class DataIngestion:
    def __init__(self):
        log.info("Reading config.yaml file")
        self.var = read_yml("config/config.yaml")
        self.raw_path = self.var['data_dir']['raw_data']

    def read_data(self) -> DataFrame:
        self.df = pd.read_csv(self.raw_path)
        return self.df


# if __name__=='__main__':
#     obj = DataIngestion()
#     obj.read_data()