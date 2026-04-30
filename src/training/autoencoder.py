from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader, TensorDataset

class LitAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def main() -> None:
    mlflow.set_experiment("Network-IDS")
    with mlflow.start_run(run_name="autoencoder_training"):
        artifacts_dir = "../../artifacts"
        data_dir = os.path.join(artifacts_dir, "data")
        
        with open(os.path.join(artifacts_dir, "label_map.json"), "r") as f:
            label_map = json.load(f)
        benign_idx = label_map.get("Benign", 0)

        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))

        X_train_benign = X_train[y_train == benign_idx]
        X_val_benign = X_val[y_val == benign_idx]

        train_tensor = torch.tensor(X_train_benign, dtype=torch.float32)
        val_tensor = torch.tensor(X_val_benign, dtype=torch.float32)

        dataset = TensorDataset(train_tensor, train_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        input_dim = X_train.shape[1]
        model = LitAutoencoder(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        mlflow.log_params({"epochs": 15, "batch_size": 256, "learning_rate": 1e-3, "input_dim": input_dim})

        print("Training Autoencoder...")
        for epoch in range(15):
            model.train()
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss/len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor)
            mses = torch.mean((val_pred - val_tensor)**2, dim=1).numpy()
            tau = float(np.percentile(mses, 99.9))
        
        mlflow.log_metric("anomaly_threshold_tau", tau)
        print(f"Calculated Anomaly Threshold (Tau): {tau:.4f}")

        with open(os.path.join(artifacts_dir, "ae_threshold.json"), "w") as f:
            json.dump({"tau": tau}, f)

        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, os.path.join(artifacts_dir, "autoencoder.pt"))
        
        mlflow.log_artifact(os.path.join(artifacts_dir, "ae_threshold.json"))
        mlflow.pytorch.log_model(model, "autoencoder_model")
        print("Autoencoder artifacts saved!")

if __name__ == "__main__":
    main()