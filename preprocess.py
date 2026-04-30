"""
CICIDS2017 Preprocessing Pipeline
Team Accelerators — ML-IDS Project

Run from your project root:
    python preprocess.py

Output:
    data/processed/X_train.csv
    data/processed/X_val.csv
    data/processed/X_test.csv
    data/processed/y_train.csv
    data/processed/y_val.csv
    data/processed/y_test.csv
    data/processed/X_normal_only.csv  <- Autoencoder training only
    data/processed/label_encoder.pkl
    data/processed/scaler.pkl
"""

import os
import re
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────

RAW_DIR          = "data/raw"
PROCESSED_DIR    = "data/processed"
RANDOM_STATE     = 42
USE_ALL_FEATURES = False

IMPORTANT_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Fwd PSH Flags',
    'Fwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Idle Mean',
    'Idle Std',
]


# ── LABEL CLEANER ──────────────────────────────────────────────────────────

def clean_label(raw_label):
    """
    Maps a raw label to a clean class name.
    Uses regex for Web Attack labels to handle ALL encoding variants:
    \x96, ï¿½, â€", \u2013 — all are matched with .+ wildcard.
    """
    s = str(raw_label).strip()

    # Web Attacks — regex handles every encoding variant
    if re.search(r'Web Attack.+Brute Force', s, re.IGNORECASE):
        return 'WebAttack'
    if re.search(r'Web Attack.+XSS', s, re.IGNORECASE):
        return 'WebAttack'
    if re.search(r'Web Attack.+Sql', s, re.IGNORECASE):
        return 'WebAttack'

    # Everything else — exact match
    exact = {
        'BENIGN'          : 'BENIGN',
        'DoS Hulk'        : 'DoS',
        'DoS GoldenEye'   : 'DoS',
        'DoS slowloris'   : 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DDoS'            : 'DDoS',
        'FTP-Patator'     : 'BruteForce',
        'SSH-Patator'     : 'BruteForce',
        'Bot'             : 'Botnet',
        'Infiltration'    : 'Infiltration',
        'PortScan'        : 'PortScan',
        'Heartbleed'      : 'Heartbleed',
    }
    return exact.get(s, None)


# ── STEP 1: LOAD ───────────────────────────────────────────────────────────

def load_all_csvs(raw_dir):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"\n[ERROR] No CSV files found in '{raw_dir}/'\n"
            f"Make sure your CICIDS2017 CSV files are inside that folder.\n"
        )

    print(f"\n{'='*60}")
    print(f"  CICIDS2017 Preprocessing — Team Accelerators")
    print(f"{'='*60}")
    print(f"\n[1/6] Loading {len(csv_files)} CSV files...\n")

    dfs = []
    for f in sorted(csv_files):
        name = os.path.basename(f)
        # Try multiple encodings — handles both Windows and Linux saved files
        df = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(f, encoding=enc, low_memory=False)
                df.columns = df.columns.str.strip()
                break
            except Exception:
                continue
        if df is None:
            print(f"  [SKIP] Could not read: {name}")
            continue
        print(f"  + {name:<58} {len(df):>8,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined):,}")
    return combined


# ── STEP 2: CLEAN ──────────────────────────────────────────────────────────

def clean(df):
    print(f"\n[2/6] Cleaning data...")
    original_len = len(df)

    # Fix label column with leading space (common in CICIDS2017)
    if ' Label' in df.columns:
        df.rename(columns={' Label': 'Label'}, inplace=True)

    # Drop rows with no label
    df = df.dropna(subset=['Label'])

    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  Found {missing:,} missing/inf values — dropping affected rows")
    df.dropna(inplace=True)

    # Remove exact duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"  Found {dupes:,} duplicate rows — removing")
        df.drop_duplicates(inplace=True)

    print(f"  Removed {original_len - len(df):,} bad rows")
    print(f"  Clean rows remaining: {len(df):,}")
    return df


# ── STEP 3: LABELS ─────────────────────────────────────────────────────────

def process_labels(df):
    print(f"\n[3/6] Processing labels...")

    print("\n  Raw label counts:")
    for label, count in df['Label'].value_counts().items():
        print(f"    {str(label):<45} {count:>8,}")

    df['Label'] = df['Label'].str.strip()

    # Apply regex-based clean_label function
    df['Label_clean'] = df['Label'].apply(clean_label)

    # Warn about anything still unmapped
    unmapped = df[df['Label_clean'].isna()]['Label'].unique()
    if len(unmapped) > 0:
        print(f"\n  [WARN] Still unmapped labels — treating as UNKNOWN:")
        for u in unmapped:
            print(f"    repr: {repr(u)}")
        df['Label_clean'] = df['Label_clean'].fillna('UNKNOWN')
    else:
        print(f"\n  All labels mapped successfully — no UNKNOWN labels!")

    # Integer encode
    le = LabelEncoder()
    df['Label_encoded'] = le.fit_transform(df['Label_clean'])

    print(f"\n  Clean label distribution:")
    for label, count in df['Label_clean'].value_counts().items():
        pct = count / len(df) * 100
        bar = chr(9608) * int(pct / 2)
        print(f"    {label:<20} {count:>8,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Label encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"    {i} = {cls}")

    return df, le


# ── STEP 4: FEATURES ───────────────────────────────────────────────────────

def select_features(df):
    print(f"\n[4/6] Selecting features...")

    if USE_ALL_FEATURES:
        drop_cols    = ['Label', 'Label_clean', 'Label_encoded']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        print(f"  Using all {len(feature_cols)} features")
    else:
        available     = [f for f in IMPORTANT_FEATURES if f in df.columns]
        missing_feats = [f for f in IMPORTANT_FEATURES if f not in df.columns]
        if missing_feats:
            print(f"  [WARN] {len(missing_feats)} features not found:")
            for m in missing_feats:
                print(f"    - {m}")
        feature_cols = available
        print(f"  Using {len(feature_cols)} selected features")

    return df, feature_cols


# ── STEP 5: SCALE ──────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    print(f"\n[5/6] Scaling features (MinMaxScaler)...")
    scaler = MinMaxScaler()

    # Fit ONLY on training data — never on val/test (prevents data leakage)
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_       train.columns
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    print(f"  Scaler fit on training data only (no data leakage)")
    return X_train_s, X_val_s, X_test_s, scaler


# ── STEP 6: SPLIT & SAVE ───────────────────────────────────────────────────

def split_and_save(df, feature_cols, le, processed_dir):
    print(f"\n[6/6] Splitting and saving...")
    os.makedirs(processed_dir, exist_ok=True)

    X = df[feature_cols]
    y = df['Label_encoded']

    # Stratified split: 70% train | 15% val | 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Reset indices after split — fixes pandas IndexingError
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # Scale
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # Save all splits as CSV
    print(f"\n  Saving CSV files...")
    X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
    X_val.to_csv(  f"{processed_dir}/X_val.csv",   index=False)
    X_test.to_csv( f"{processed_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
    y_val.to_csv(  f"{processed_dir}/y_val.csv",   index=False)
    y_test.to_csv( f"{processed_dir}/y_test.csv",  index=False)

    # Save normal-only data for Autoencoder
    # CRITICAL: Autoencoder must ONLY ever see clean BENIGN traffic
    normal_label = le.transform(['BENIGN'])[0]
    X_normal = X_train[y_train == normal_label].reset_index(drop=True)
    X_normal.to_csv(f"{processed_dir}/X_normal_only.csv", index=False)
    print(f"  Normal-only samples for Autoencoder: {len(X_normal):,}")

    # Save scaler + label encoder for inference pipeline
    with open(f"{processed_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{processed_dir}/label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Train samples : {len(X_train):>8,}")
    print(f"  Val samples   : {len(X_val):>8,}")
    print(f"  Test samples  : {len(X_test):>8,}")
    print(f"  Features used : {len(feature_cols):>8}")
    print(f"  Classes       : {len(le.classes_):>8}")
    print(f"\n  Files saved to '{processed_dir}/':")
    files = [
        'X_train.csv', 'X_val.csv', 'X_test.csv',
        'y_train.csv', 'y_val.csv', 'y_test.csv',
        'X_normal_only.csv', 'scaler.pkl', 'label_encoder.pkl'
    ]
    for fn in files:
        path = f"{processed_dir}/{fn}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    - {fn:<35} {size_mb:>7.1f} MB")
    print(f"{'='*60}\n")


# ── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df               = load_all_csvs(RAW_DIR)
    df               = clean(df)
    df, le           = process_labels(df)
    df, feature_cols = select_features(df)
    split_and_save(df, feature_cols, le, PROCESSED_DIR)