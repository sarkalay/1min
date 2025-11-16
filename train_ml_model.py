# train_ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

MODEL_FILE = "sl_mistake_classifier.pkl"
DATA_FILE = "sl_analysis_dataset.csv"

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] No data file: {DATA_FILE}")
        print("Run some trades with log_trade_for_ml() first.")
        return False
    
    df = pd.read_csv(DATA_FILE)
    if len(df) < 20:
        print(f"[WARNING] Only {len(df)} samples. Need at least 20 for training.")
        return False
    
    print(f"[TRAIN] Loaded {len(df)} trades. Training model...")
    
    # Features
    feature_cols = [col for col in df.columns if col not in ["is_mistake", "timestamp", "pair"]]
    X = df[feature_cols]
    y = df["is_mistake"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n" + classification_report(y_test, preds, target_names=["Normal", "Mistake"]))
    
    joblib.dump(model, MODEL_FILE)
    print(f"[SUCCESS] Model saved: {MODEL_FILE}")
    return True

if __name__ == "__main__":
    train_model()
