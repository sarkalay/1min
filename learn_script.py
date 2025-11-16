# learn_script.py
import os
import json
import time
from data_collector import log_trade_for_ml
from ml_predictor import SLPredictor

class SelfLearningAITrader:
    def __init__(self):
        # === FILES ===
        self.mistakes_history_file = "ai_trading_mistakes.json"
        self.learned_patterns_file = "ai_learned_patterns.json"
        
        # === LOAD HISTORY ===
        self.mistakes_history = self.load_mistakes_history()
        self.learned_patterns = self.load_learned_patterns()
        
        # === STATS ===
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'common_mistakes': {},
            'improvement_areas': []
        }
        
        # === CONFIG ===
        self.learning_config = {
            'confidence_threshold': 0.7,
            'min_trades_to_learn': 3
        }
        
        # === ML PREDICTOR ===
        self.ml_predictor = SLPredictor()
        print(f"[AI] Self-Learning System Ready | Mistakes: {len(self.mistakes_history)} | Patterns: {len(self.learned_patterns)}")
    
    def load_mistakes_history(self):
        try:
            if os.path.exists(self.mistakes_history_file):
                with open(self.mistakes_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except: pass
        return []
    
    def save_mistakes_history(self):
        try:
            with open(self.mistakes_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.mistakes_history, f, indent=2, ensure_ascii=False)
        except: pass

    def load_learned_patterns(self):
        try:
            if os.path.exists(self.learned_patterns_file):
                with open(self.learned_patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except: pass
        return {}

    def save_learned_patterns(self):
        try:
            with open(self.learned_patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_patterns, f, indent=2, ensure_ascii=False)
        except: pass

    def analyze_trade_mistake(self, trade_data):
        """အမှား အမျိုးအစား ခွဲခြားမယ်"""
        if trade_data.get("pnl", 0) >= 0:
            return None
        
        direction = trade_data["direction"]
        close_reason = trade_data.get("close_reason", "")
        entry_price = trade_data["entry_price"]
        exit_price = trade_data["exit_price"]
        
        loss_pct = abs((exit_price - entry_price) / entry_price * 100) if direction == "LONG" else abs((entry_price - exit_price) / entry_price * 100)
        
        mistake_type = "UNKNOWN"
        lesson = ""
        avoidance = ""
        
        if "STOP_LOSS" in close_reason.upper():
            mistake_type = f"{direction} stopped out"
            lesson = "Entry in low-probability setup"
            avoidance = "Wait for stronger confirmation + proper risk"
        elif "TREND_REVERSAL" in close_reason:
            mistake_type = f"{direction} against trend"
            lesson = "Do not fight the trend"
            avoidance = "Confirm 1H/4H alignment"
        elif "OVERSOLD" in close_reason or "OVERBOUGHT" in close_reason:
            mistake_type = f"{direction} in extreme RSI"
            lesson = "Avoid chasing overextended moves"
            avoidance = "Wait for pullback"
        
        return {
            "mistake_type": mistake_type,
            "lesson_learned": lesson,
            "avoidance_strategy": avoidance,
            "trade_data": trade_data,
            "pnl": trade_data["pnl"],
            "loss_percent": round(loss_pct, 2),
            "timestamp": time.time()
        }

    def learn_from_mistake(self, trade_data, market_data=None, force_mistake=None):
        """
        force_mistake: True/False/None
            - True  → လူက အတင်း အမှား လို့ သတ်မှတ်
            - False → လူက မဟုတ်ဘူး လို့ သတ်မှတ်
            - None  → ML က ဆုံးဖြတ်
        """
        if market_data is None:
            market_data = {}
        
        close_reason = trade_data.get("close_reason", "")
        is_sl_hit = "STOP_LOSS" in close_reason.upper() if close_reason else False
        
        # === ML က ဆုံးဖြတ်မယ် (သို့မဟုတ် force) ===
        if is_sl_hit and force_mistake is None:
            is_mistake = self.ml_predictor.predict_mistake(trade_data, market_data)
        else:
            is_mistake = force_mistake if force_mistake is not None else (trade_data.get("pnl", 0) < 0)
        
        # === DATA LOG FOR ML TRAINING ===
        log_trade_for_ml(trade_data, market_data, is_mistake=is_mistake)
        
        # === အမှား မဟုတ်ရင် skip ===
        if not is_mistake:
            print(f"[LEARN] SL hit but NOT a mistake → Skipping learning")
            return
        
        print(f"[LEARN] Confirmed mistake → Analyzing...")
        
        # === အမှား ခွဲခြားပြီး သင်ယူမယ် ===
        analysis = self.analyze_trade_mistake(trade_data)
        if analysis:
            self.mistakes_history.append(analysis)
            self.update_learned_patterns(analysis)
            self.save_mistakes_history()
            self.save_learned_patterns()
            print(f"[LEARN] Lesson saved: {analysis['lesson_learned']}")

    def update_learned_patterns(self, analysis):
        """Pattern တွေ မှတ်မိအောင် လုပ်မယ်"""
        mistake_type = analysis["mistake_type"]
        if mistake_type not in self.learned_patterns:
            self.learned_patterns[mistake_type] = {
                "count": 0,
                "total_loss": 0,
                "avoidance": analysis["avoidance_strategy"]
            }
        self.learned_patterns[mistake_type]["count"] += 1
        self.learned_patterns[mistake_type]["total_loss"] += abs(analysis["pnl"])

    def should_avoid_trade(self, ai_decision, market_data):
        """AI ရဲ့ decision ကို သင်ယူထားတဲ့ pattern နဲ့ စစ်မယ်"""
        if ai_decision["decision"] in ["HOLD", "REVERSE_LONG", "REVERSE_SHORT"]:
            return False
        
        direction = ai_decision["decision"]
        pair = ai_decision.get("pair", "UNKNOWN")
        
        # ဥပမာ: LONG လုပ်မယ်ဆို → အရင် LONG မှာ အမှားများလား?
        pattern_key = f"{direction} stopped out"
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            if pattern["count"] >= 3:
                print(f"[BLOCK] Avoiding {direction} on {pair} - Known mistake pattern ({pattern['count']} times)")
                return True
        return False

    def get_learning_enhanced_prompt(self, pair, market_data):
        """AI Prompt ထဲ သင်ယူထားတာ ထည့်ပေးမယ်"""
        if not self.mistakes_history:
            return ""
        
        recent = self.mistakes_history[-3:]
        lessons = []
        for m in recent:
            lessons.append(f"- {m['lesson_learned']} (Loss: ${abs(m['pnl']):.2f})")
        
        return f"""
LEARNING CONTEXT (from {len(self.mistakes_history)} past mistakes):
{chr(10).join(lessons)}
Apply these lessons to avoid repeating errors.
"""

    def adaptive_learning_adjustment(self):
        """Performance ပေါ် မူတည်ပြီး parameter တွေ ပြောင်းမယ်"""
        if self.performance_stats['total_trades'] == 0:
            return
        
        win_rate = self.performance_stats['winning_trades'] / self.performance_stats['total_trades']
        if win_rate < 0.4:
            print(f"[ADAPT] Win rate low ({win_rate:.1%}) → Increasing caution")
            # ဥပမာ: position size လျှော့မယ်
        elif win_rate > 0.7:
            print(f"[ADAPT] Win rate high ({win_rate:.1%}) → Increasing confidence")
