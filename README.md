#  Hybrid Network Intrusion Detection System (IDS)


An end-to-end machine learning pipeline for detecting network intrusions using a **hybrid approach** that combines supervised learning for known attacks and anomaly detection for identifying unknown or zero-day threats.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
- [Model Performance](#model-performance)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Traditional IDS systems rely on predefined rules and fail to detect new attack patterns. This project addresses that limitation by implementing a **hybrid detection approach**:

| Component | Purpose | Method |
|-----------|---------|--------|
| **XGBoost Classifier** | Detects known attacks | Supervised Learning |
| **Autoencoder** | Detects anomalies/unseen attacks | Unsupervised Learning |

The final system combines both outputs to improve detection accuracy and reduce false positives, making it robust against both known and zero-day attacks.

---

## ✨ Features

- **Hybrid Detection**: Combines supervised and unsupervised learning
- **Handles Noisy Data**: Robust preprocessing for real-world network traffic
- **Scalable Pipeline**: End-to-end automated workflow
- **Model Persistence**: Saves trained models for production deployment
- **Zero-Day Detection**: Identifies previously unseen attack patterns
- **Low False Positives**: Dual-model approach reduces incorrect alerts

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Network Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│  - Cleaning     │
│  - Scaling      │
└────────┬────────┘
         │
         ▼
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────┐   ┌──────────┐
│ XGB │   │Autoencoder│
│Model│   │  (Anomaly)│
└──┬──┘   └─────┬────┘
   │            │
   └─────┬──────┘
         ▼
   ┌──────────┐
   │  Hybrid  │
   │ Decision │
   │  (OR)    │
   └─────┬────┘
         ▼
   ┌──────────┐
   │ Attack?  │
   │ Yes/No   │
   └──────────┘
```

**Hybrid Decision Rule:**
```python
Final Prediction = XGBoost Prediction OR Autoencoder Anomaly
```

---

## 📊 Dataset

**[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)**

- Network flow-based dataset with **70+ features**
- Realistic network traffic with labeled attacks
- Multiple attack categories:
  - DoS / DDoS
  - Port Scanning
  - Web Attacks
  - Brute Force
  - Infiltration
  - Botnet

---

## 📁 Project Structure

```
ml-ids/
├─ docker/
│  ├─ docker-compose.yml
│  └─ k8s/                       # optional k8s manifests
├─ data/
│  ├─ raw/                       # downloaded datasets (NSL-KDD, UNSW-NB15, CICIDS2017)
│  └─ processed/
├─ notebooks/
│  └─ eda_and_feature_engineering.ipynb
├─ src/
│  ├─ preprocessing/
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ autoencoder.py
│  │  └─ xgb_trainer.py
│  ├─ inference/
│  │  ├─ app/
│  │  │  ├─ main.py
│  │  │  ├─ model_loader.py
│  │  │  └─ requirements.txt
│  │  └─ kafka_consumer.py
│  ├─ producer/
│  │  └─ simulate_traffic.py
│  ├─ mlops/
│  │  ├─ celery_worker.py
│  │  └─ tasks.py
│  └─ utils/
│     └─ feature_utils.py
├─ deployments/
│  ├─ fastapi-deployment.yaml
│  ├─ mlflow-deployment.yaml
│  └─ kafka-deployment.yaml
├─ Dockerfile (for training & inference images)
└─ README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml_ids_project.git
   cd ml_ids_project
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy scikit-learn xgboost torch joblib
   ```

4. **Download dataset**
   - Download CIC-IDS2017 from [official source](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Place CSV files in `data/raw/` directory

---

## 💻 Usage

### Training the Model

Run the complete pipeline:

```bash
python pipeline.py
```

This will:
1. Load and merge all CSV files from `data/raw/`
2. Clean and preprocess the data
3. Train both XGBoost and Autoencoder models
4. Save all artifacts to `artifacts/` directory

### Output

After successful execution, the following files will be generated:

```
artifacts/
├── scaler.joblib                 # Feature scaler
├── xgb_model.joblib             # XGBoost model
├── autoencoder.pt               # Autoencoder weights
└── autoencoder_threshold.npy    # Anomaly threshold
```

### Using Trained Models

```python
import joblib
import torch
import numpy as np

# Load artifacts
scaler = joblib.load('artifacts/scaler.joblib')
xgb_model = joblib.load('artifacts/xgb_model.joblib')
autoencoder = torch.load('artifacts/autoencoder.pt')
threshold = np.load('artifacts/autoencoder_threshold.npy')

# Predict on new data
X_new_scaled = scaler.transform(X_new)

# XGBoost prediction
xgb_pred = xgb_model.predict(X_new_scaled)

# Autoencoder anomaly detection
reconstruction = autoencoder(torch.FloatTensor(X_new_scaled))
error = np.mean((X_new_scaled - reconstruction.detach().numpy())**2, axis=1)
ae_pred = (error > threshold).astype(int)

# Hybrid decision
final_prediction = np.logical_or(xgb_pred, ae_pred).astype(int)
```

---

## 🔄 Pipeline Details

### Step 1: Data Loading
- Merges multiple CSV files from the dataset
- Handles large files efficiently using chunking

### Step 2: Data Cleaning
- Removes missing values (`NaN`)
- Filters infinite values (`inf`, `-inf`)
- Ensures data quality for training

### Step 3: Label Processing
- Binary classification:
  - `BENIGN` → **0** (Normal traffic)
  - All attack types → **1** (Malicious traffic)

### Step 4: Feature Scaling
- Applies `StandardScaler` for normalization
- Ensures features are on the same scale
- Critical for both XGBoost and neural networks

### Step 5: Model Training

**XGBoost Classifier:**
- Trained on labeled data (supervised)
- Learns patterns of known attacks
- Fast inference, high accuracy

**Autoencoder:**
- Trained only on normal traffic (unsupervised)
- Learns to reconstruct benign patterns
- High reconstruction error indicates anomaly

### Step 6: Anomaly Detection
- Calculates reconstruction error for each sample
- Sets threshold based on normal traffic distribution
- Flags samples exceeding threshold as anomalies

### Step 7: Hybrid Decision
```
IF (XGBoost predicts ATTACK) OR (Autoencoder detects ANOMALY):
    ALERT: Intrusion Detected
ELSE:
    BENIGN: Normal Traffic
```

### Step 8: Model Persistence
- Saves all trained models and preprocessing objects
- Enables deployment without retraining
- Version control for model artifacts

---

## 📈 Model Performance

| Metric | XGBoost | Autoencoder | Hybrid |
|--------|---------|-------------|--------|
| Accuracy | High | Medium | **Highest** |
| False Positives | Low | Medium | **Lowest** |
| Zero-Day Detection | ❌ | ✅ | ✅ |
| Known Attack Detection | ✅ | Medium | ✅ |

*Note: Exact metrics depend on dataset split and hyperparameters*

---

## 🔮 Future Work

### Experiment Tracking
- [ ] Integrate **MLflow** for experiment management
- [ ] Track hyperparameters, metrics, and model versions
- [ ] Enable model comparison and A/B testing

### Deployment
- [ ] Build REST API using **FastAPI**
- [ ] Containerize with **Docker**
- [ ] Deploy to cloud (AWS, GCP, Azure)

### Real-Time Processing
- [ ] Integrate **Kafka** for streaming data
- [ ] Implement sliding window detection
- [ ] Add real-time alerting system

### Enhanced Detection
- [ ] Extend to **multi-class classification** (classify attack types)
- [ ] Implement **ensemble methods** (stacking, voting)
- [ ] Add **explainability** with SHAP values

### Production Features
- [ ] Add monitoring and logging
- [ ] Implement model retraining pipeline
- [ ] Create dashboard for visualization
- [ ] Add data drift detection

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---



## 🙏 Acknowledgments

- **Canadian Institute for Cybersecurity** for the CIC-IDS2017 dataset
- XGBoost and PyTorch communities for excellent documentation
- Open-source contributors

---

## 📧 Contact

For questions or support, please open an issue 

---

<div align="center">
  
**⭐ Star this repository if you find it helpful!**

</div>
