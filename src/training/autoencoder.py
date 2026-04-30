from __future__ import annotations

import json
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LitAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def main() -> None:
    artifacts_dir = "../../artifacts"
    data_dir = os.path.join(artifacts_dir, "data")
    
    # Load label map to find Benign class integer
    with open(os.path.join(artifacts_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    benign_idx = label_map.get("Benign", 0)

    # Load data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    # Isolate benign traffic
    X_train_benign = X_train[y_train == benign_idx]
    X_val_benign = X_val[y_val == benign_idx]

    train_tensor = torch.tensor(X_train_benign, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_benign, dtype=torch.float32)

    dataset = TensorDataset(train_tensor, train_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = X_train.shape[1]
    model = LitAutoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training Autoencoder...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # Determine threshold (tau) using 95th percentile of validation benign error
    model.eval()
    with torch.no_grad():
        val_pred = model(val_tensor)
        mses = torch.mean((val_pred - val_tensor)**2, dim=1).numpy()
        tau = np.percentile(mses, 95)
    
    print(f"Calculated Anomaly Threshold (Tau): {tau:.4f}")

    # Save threshold and PyTorch JIT model
    with open(os.path.join(artifacts_dir, "ae_threshold.json"), "w") as f:
        json.dump({"tau": float(tau)}, f)

    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(artifacts_dir, "autoencoder.pt"))
    print("Autoencoder artifacts saved!")

if __name__ == "__main__":
    main()