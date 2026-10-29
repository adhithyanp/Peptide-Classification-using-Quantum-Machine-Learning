# models/classical/svm_model.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)

        metrics = {
            "model": "SVM",
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }

        return metrics

    def predict(self, X):
        return self.model.predict(X)
