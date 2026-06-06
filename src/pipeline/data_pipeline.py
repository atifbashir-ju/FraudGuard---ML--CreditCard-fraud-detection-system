"""
src/pipeline/data_pipeline.py
Handles data loading, cleaning, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

RAW_PATH       = os.path.join('data', 'raw',       'creditcard.csv')
PROCESSED_PATH = os.path.join('data', 'processed', 'features.csv')
SCALER_PATH    = os.path.join('models',             'scaler.pkl')


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows | Fraud rate: {df['Class'].mean()*100:.2f}%")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (30+ features)
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Time features
    df['hour']           = (df['Time'] % 86400) // 3600
    df['is_night']       = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_weekend']     = ((df['Time'] // 86400) % 7 >= 5).astype(int)

    # -- Amount features
    df['log_amount']     = np.log1p(df['Amount'])
    df['amount_squared'] = df['Amount'] ** 2
    df['is_round_amt']   = (df['Amount'] % 10 == 0).astype(int)
    df['is_small_amt']   = (df['Amount'] < 5).astype(int)
    df['is_large_amt']   = (df['Amount'] > 500).astype(int)

    # -- V-feature interactions (top correlated with fraud)
    df['v1_v2']          = df['V1'] * df['V2']
    df['v3_v4']          = df['V3'] * df['V4']
    df['v1_v3']          = df['V1'] * df['V3']
    df['v7_v20']         = df['V7'] * df['V20']
    df['v14_v17']        = df['V14'] * df['V17']

    # -- Statistical aggregates on V features
    v_cols               = [f'V{i}' for i in range(1, 29)]
    df['v_mean']         = df[v_cols].mean(axis=1)
    df['v_std']          = df[v_cols].std(axis=1)
    df['v_max']          = df[v_cols].max(axis=1)
    df['v_min']          = df[v_cols].min(axis=1)
    df['v_range']        = df['v_max'] - df['v_min']

    # -- Risk score (domain knowledge)
    df['risk_score']     = (
        -df['V14'] * 0.3
        - df['V17'] * 0.2
        + df['V12'] * 0.15
        + df['log_amount'] * 0.1
        + df['is_night'] * 0.25
    )

    print(f"[INFO] Feature engineering done — {df.shape[1]} total columns")
    return df


# ─────────────────────────────────────────────
# 3. CLEAN & VALIDATE
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.dropna()
    df     = df.drop_duplicates()
    print(f"[INFO] Cleaned: {before - len(df):,} rows removed")
    return df


# ─────────────────────────────────────────────
# 4. SCALE
# ─────────────────────────────────────────────
def scale_features(X_train, X_test, cols_to_scale):
    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved → {SCALER_PATH}")
    return X_train, X_test, scaler


# ─────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(path: str = RAW_PATH):
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    DROP_COLS = ['Time']
    TARGET    = 'Class'

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_cols = ['Amount', 'log_amount', 'amount_squared', 'risk_score',
                  'v_mean', 'v_std', 'v_max', 'v_min', 'v_range']
    scale_cols = [c for c in scale_cols if c in X_train.columns]

    X_train, X_test, scaler = scale_features(X_train, X_test, scale_cols)

    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    run_pipeline()
