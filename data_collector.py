# data_collector.py
import csv
import os
import time

DATA_FILE = "sl_analysis_dataset.csv"

def log_trade_for_ml(trade_data, market_data, is_mistake=False):
    """
    Trade တစ်ခုကို ML dataset ထဲ သိမ်းမယ်
    is_mistake = True → လူက အမှားလို့ သတ်မှတ်ထားတာ
    """
    row = {
        "timestamp": trade_data.get("close_timestamp", time.time()),
        "pair": trade_data["pair"],
        "direction": 1 if trade_data["direction"] == "LONG" else 0,
        "entry_price": trade_data["entry_price"],
        "exit_price": trade_data["exit_price"],
        "pnl": trade_data["pnl"],
        "leverage": trade_data.get("leverage", 1),
        "position_size_usd": trade_data.get("position_size_usd", 50.0),
        "loss_percent": abs(trade_data["pnl"]) / trade_data.get("position_size_usd", 50.0) * 100,
        "atr_percent": market_data.get("atr_percent", 0),
        "volatility_spike": 1 if market_data.get("atr_percent", 0) > 3.0 else 0,
        "trend_strength": market_data.get("trend_strength", 0),  # -1 to +1
        "rsi": market_data.get("rsi", 50),
        "volume_change": market_data.get("volume_change", 0),
        "news_impact": 1 if market_data.get("news_impact", False) else 0,
        "sl_distance_pct": market_data.get("sl_distance_pct", 0),
        "is_mistake": 1 if is_mistake else 0
    }
    
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    status = "MISTAKE" if is_mistake else "NORMAL"
    print(f"[DATA] Logged: {trade_data['pair']} | {status} | PnL: {trade_data['pnl']:.2f}")
