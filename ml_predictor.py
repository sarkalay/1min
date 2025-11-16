# ml_predictor.py
import joblib
import pandas as pd
import os

MODEL_FILE = "sl_mistake_classifier.pkl"

class SLPredictor:
    def __init__(self):
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
            print(f"[ML] Model loaded: {MODEL_FILE}")
        else:
            self.model = None
            print(f"[ML] No model found. Using fallback rules.")
    
    def predict_mistake(self, trade_data, market_data):
        if not self.model:
            # Fallback rule
            return "STOP_LOSS" in trade_data.get("close_reason", "").upper()
        
        input_data = {
            "direction": 1 if trade_data["direction"] == "LONG" else 0,
            "entry_price": trade_data["entry_price"],
            "exit_price": trade_data["exit_price"],
            "pnl": trade_data["pnl"],
            "leverage": trade_data.get("leverage", 1),
            "position_size_usd": trade_data.get("position_size", 50.0),
            "loss_percent": abs(trade_data["pnl"]) / trade_data.get("position_size_usd", 50.0) * 100,
            "atr_percent": market_data.get("atr_percent", 0),
            "volatility_spike": 1 if market_data.get("atr_percent", 0) > 3.0 else 0,
            "trend_strength": market_data.get("trend_strength", 0),
            "rsi": market_data.get("rsi", 50),
            "volume_change": market_data.get("volume_change", 0),
            "news_impact": 1 if market_data.get("news_impact", False) else 0,
            "sl_distance_pct": market_data.get("sl_distance_pct", 0),
        }
        
        df = pd.DataFrame([input_data])
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0][pred]
        
        result = "MISTAKE" if pred else "NORMAL"
        print(f"[PREDICT] → {result} (Confidence: {prob:.1%})")
        return bool(pred)
