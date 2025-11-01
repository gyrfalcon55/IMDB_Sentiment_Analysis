from IMDB_Project.utils.common import read_csv
from IMDB_Project.components.data_ingestion import DataIngestion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from IMDB_Project.logger import log


class ModelEvaluator:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.test_df = read_csv(self.ingestion.var['data_dir']['test_data'])
        log.info("Loaded trained model and vectorizer for evaluation")
        self.model = joblib.load(self.ingestion.var['artifacts']['model'])
        self.vectorizer = joblib.load(self.ingestion.var['artifacts']['vectorizer'])

    def evaluate(self):
        log.info('Saved model evaluation')
        x_test_vec = self.vectorizer.transform(self.test_df['text'])
        y_true = self.test_df['label']
        y_pred = self.model.predict(x_test_vec)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        log.info(f"Test Accuracy: {acc:.4f}")
        log.info(f"\nConfusion Matrix:\n{cm}")
        log.info(f"Classification Report:\n{report}")

        log.info(f"Model Evaluation Complete\nAccuracy: {acc:.4f}\n")


# if __name__ == "__main__":
#     evaluator = ModelEvaluator()
#     evaluator.evaluate()
