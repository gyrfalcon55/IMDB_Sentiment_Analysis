import yaml
from pandas import DataFrame
import pandas as pd

def read_yml(yaml_path) -> dict:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def print_data(data_set):
    if isinstance(data_set, str):  # If a path is passed
        data_set = pd.read_csv(data_set)
    print(data_set.head(5))


def read_csv(csv_path) -> DataFrame:
    df = pd.read_csv(csv_path)
    return df