from IMDB_Project.logger import log
from IMDB_Project.components.data_ingestion import DataIngestion
from IMDB_Project.utils.common import read_yml,print_data,read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,f1_score
import numpy as np
import joblib
import importlib
import os

import mlflow
import dagshub

dagshub.init(repo_owner='gyrfalcon55', repo_name='IMDB_Sentiment_Analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/gyrfalcon55/IMDB_Sentiment_Analysis.mlflow")
mlflow.set_experiment("IMDB_Sentiment_Analysis_v1")




class Preparing_Data:
    def __init__(self):
        self.ingestion = DataIngestion()
        log.info("Train and Test Data loaded")
        self.train_df = read_csv(self.ingestion.var['data_dir']['train_data'])
        self.test_df = read_csv(self.ingestion.var['data_dir']['test_data'])

    def splitting(self):
        self.x_train = self.train_df[['text']]
        self.y_train = self.train_df['label']


        self.x_test = self.test_df[['text']]
        self.y_test = self.test_df['label']

    def text_to_vectors(self):
        self.counter = CountVectorizer(max_features=10000, stop_words='english')  # optional limit
        self.x_train_counter = self.counter.fit_transform(self.x_train['text'])
        self.x_test_counter = self.counter.transform(self.x_test['text'])

class ModelTraining(Preparing_Data):

    def __init__(self):
        super().__init__()
        model_config_path = self.ingestion.var['model_config_path']
        self.model_config = read_yml(model_config_path)
        self.model_config = self.model_config['models']


    def get_class_from_string(self,class_path):
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def Hyperparameter_Tuning(self) -> dict:
        log.info("Hyperparameter Tuning Started")
        self.models_params = {}
        for model_name, model_info in self.model_config.items():
            ModelClass = self.get_class_from_string(model_info["class"])
            self.models_params[model_name] = {
                "model": ModelClass(),
                "params": model_info["params"]
            }

        self.best_overall_score = 0.0
        self.best_models = {}

        for model_name, mp in self.models_params.items():
            log.info(f"----- Running RandomizedSearchCV for {model_name} ----------")

            with mlflow.start_run(run_name=f"{model_name}_Experiment"):
                random_search = RandomizedSearchCV(
                    estimator=mp['model'],
                    param_distributions=mp['params'],
                    n_iter=5,
                    scoring='accuracy',
                    cv=3,
                    verbose=2,
                    random_state=42,
                    n_jobs=-1
                )
                random_search.fit(self.x_train_counter, self.y_train)
                y_pred = random_search.best_estimator_.predict(self.x_test_counter)
                test_acc = accuracy_score(self.y_test, y_pred)
                f1_score_ = f1_score(self.y_test, y_pred)
                roc_auc_score_ = roc_auc_score(self.y_test, y_pred)

                # Log parameters and metrics
                mlflow.log_params(random_search.best_params_)
                mlflow.log_metric("cv_score", random_search.best_score_)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("f1-score",f1_score_)
                mlflow.log_metric("roc_auc_score",roc_auc_score_)
                
                '''
                You can view or download the logged model artifact from the 
                MLflow UI under the "Artifacts" section of the run (path: models/)
                '''
                joblib.dump(random_search.best_estimator_, f"{model_name}_temp_model.pkl")
                mlflow.log_artifact(f"{model_name}_temp_model.pkl", artifact_path="models")
                os.remove(f"{model_name}_temp_model.pkl")

                log.info(f"Best params for {model_name}: {random_search.best_params_}")
                log.info(f"Best CV score: {random_search.best_score_:.4f}")
                log.info(f"Test Accuracy ({model_name}): {test_acc:.4f}")
                log.info(f"f1-score ({model_name}): {f1_score_:.4f}")
                log.info(f"roc_auc_score_ ({model_name}): {roc_auc_score_:.4f}")
                log.info(f"\n{classification_report(self.y_test, y_pred)}")

                # Save best model logic
                self.best_models[model_name] = random_search.best_estimator_
                if test_acc > self.best_overall_score:
                    self.best_overall_score = test_acc
                    self.best_model = random_search.best_estimator_
                    self.best_model_name = model_name


        log.info("Hyperparameter tuning Completed")

        os.makedirs(self.ingestion.var['artifacts']['model_dir'], exist_ok=True)
        os.makedirs(self.ingestion.var['artifacts']['vectorizer_dir'], exist_ok=True)

        log.info("Saving model and vectorizer to the files inside artifacts folder")
        joblib.dump(self.best_model, self.ingestion.var['artifacts']['model'])
        joblib.dump(self.counter, self.ingestion.var['artifacts']['vectorizer'])
        log.info(f"Best Model: {self.best_model_name} saved with Accuracy: {self.best_overall_score:.4f}")





# if __name__ == '__main__':
#     param_tuner = ModelTraining()
#     param_tuner.splitting()
#     param_tuner.text_to_vectors()
#     param_tuner.Hyperparameter_Tuning()
#     print(param_tuner.best_models)
