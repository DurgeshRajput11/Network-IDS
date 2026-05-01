from __future__ import annotations

import json
import os
import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

def main() -> None:
    mlflow.set_experiment("Network-IDS")
    with mlflow.start_run(run_name="xgboost_training"):
        artifacts_dir = "../../artifacts"
        data_dir = os.path.join(artifacts_dir, "data")

        print("Loading data for XGBoost...")
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))

        num_classes = len(np.unique(y_train))

        print("Computing sample weights for imbalanced classes...")
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softmax",
            "num_class": num_classes,
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 20  
        }
        mlflow.log_params(params)

        clf = XGBClassifier(**params)

        print("Training XGBoost with early stopping...")
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=10
        )

        print("Validating model...")
        y_pred = clf.predict(X_val)
        
        with open(os.path.join(artifacts_dir, "label_map.json"), "r") as f:
            label_map = json.load(f)
        target_names = sorted(label_map, key=label_map.get)

        bal_acc = balanced_accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        
        mlflow.log_metric("val_balanced_accuracy", bal_acc)
        mlflow.log_metric("val_macro_f1", macro_f1)

        print("\nValidation Balanced Accuracy:", bal_acc)
        print(classification_report(y_val, y_pred, target_names=target_names))

        clf.save_model(os.path.join(artifacts_dir, "xgboost_model.json"))
        
        mlflow.xgboost.log_model(clf, "xgboost_model")
        print("XGBoost training complete and saved.")

if __name__ == "__main__":
    main()