from __future__ import annotations

import json
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

def main() -> None:
    artifacts_dir = "../../artifacts"
    data_dir = os.path.join(artifacts_dir, "data")

    # Load data
    print("Loading data for XGBoost...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    num_classes = len(np.unique(y_train))

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss"
    )

    print("Training XGBoost with early stopping...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=10
    )

    print("Validating model...")
    y_pred = clf.predict(X_val)
    
    # Load label mapping to print proper classification report
    with open(os.path.join(artifacts_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    target_names = sorted(label_map, key=label_map.get)

    print("\nValidation Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=target_names))

    # Save strictly to XGBoost native JSON format
    clf.save_model(os.path.join(artifacts_dir, "xgboost_model.json"))
    print("XGBoost training complete and saved.")

if __name__ == "__main__":
    main()