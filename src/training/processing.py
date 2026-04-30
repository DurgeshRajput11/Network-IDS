from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def main() -> None:
    mlflow.set_experiment("Network-IDS")
    with mlflow.start_run(run_name="preprocessing"):
        data_path = "../../data/cic.csv"
        artifacts_dir = "../../artifacts"
        data_out_dir = os.path.join(artifacts_dir, "data")
        
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(data_out_dir, exist_ok=True)

        print("Loading data...")
        df = pd.read_csv(data_path)
        if "Timestamp" in df.columns:
            df = df.drop(columns=["Timestamp"])
        df.drop_duplicates(inplace=True)

        X = df.drop(columns=["Label"])
        y = df["Label"]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("val_size_from_temp", 0.5)

        print("Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        
        label_map = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        joblib.dump(le, os.path.join(artifacts_dir, "label_encoder.joblib"))
        with open(os.path.join(artifacts_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=4)

        print("Imputing missing values...")
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        print("Selecting top 20 features via L1 RFE...")
        estimator = LogisticRegression(penalty="l1", solver="liblinear", random_state=42, max_iter=500)
        target_features = 20
        rfe = RFE(estimator=estimator, n_features_to_select=target_features, step=0.05)
        rfe.fit(X_train_imp, y_train_enc)
        
        mlflow.log_param("rfe_features_selected", target_features)
        
        selected_features = X_train_imp.columns[rfe.support_].tolist()
        with open(os.path.join(artifacts_dir, "features_schema.json"), "w") as f:
            json.dump({"features": selected_features}, f, indent=4)
            
        X_train_sel = X_train_imp[selected_features]
        X_val_sel = X_val_imp[selected_features]
        X_test_sel = X_test_imp[selected_features]

        print("Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))

        print("Saving processed arrays...")
        np.save(os.path.join(data_out_dir, "X_train.npy"), X_train_scaled)
        np.save(os.path.join(data_out_dir, "y_train.npy"), y_train_enc)
        np.save(os.path.join(data_out_dir, "X_val.npy"), X_val_scaled)
        np.save(os.path.join(data_out_dir, "y_val.npy"), y_val_enc)
        np.save(os.path.join(data_out_dir, "X_test.npy"), X_test_scaled)
        np.save(os.path.join(data_out_dir, "y_test.npy"), y_test_enc)
        
        mlflow.log_metric("train_samples", X_train_scaled.shape[0])
        mlflow.log_artifact(os.path.join(artifacts_dir, "features_schema.json"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "scaler.joblib"))
        
        print("Preprocessing complete!")

if __name__ == "__main__":
    main()