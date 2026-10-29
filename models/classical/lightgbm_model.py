# models/classical/lightgbm_model.py

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LightGBMModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)

        return {
            "model": "LightGBM",
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }

    def predict(self, X):
        return self.model.predict(X)
